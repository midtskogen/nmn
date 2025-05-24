#!/bin/bash

# Script for stellar calibration of a lens.  Input is video showing the clear
# night sky, and the script should be able to figure out the stars and the
# precise geometry of the lens and orientation of the camera, and produce a
# .pto file which can be used by the Hugin tools to translate between x,y
# and az,alt.
#
# Example usage:
#   calibrate.sh /meteor/cam2/20141125/0[0-4]/full_*.mp4
#   calibrate.sh full_*.jpg
#   calibrate.sh 9  # Restart previous/interrupted run at stage 9
# Output:
#   out/lens2.pto
#   out/combined.mp4
#   out/grid.png
#   ... and a lot more files.
#
# A couple of hours of video is usually sufficient.  On a pretty fast PC, the
# script should finish in roughly half of the time of the videos.  If many
# hours of video is supply and/or it's noisy, stage 4 could become very slow
# and run for days without giving good results.
#
# Written for Vivotek IP8172 files, but can surely be adapted to other cameras.
# Assumes small timestamp in BL corner.  If no timestamp, out/timestamp.txt must
# be created manually, Unix time, one line for every input file.
#
# This script is a hack.  Use with caution.  It probably wont work out of the
# box for a camera different from Vivotek 8172.  If few stars are detected,
# try tweaking the solve-field arguments (stage 8) and/or the image
# preprocessing (stage 1, 2)
#
# Written by Steinar Midtskogen <steinar@norskmeteornettverk.no>.

# Multithreading
#MT=3
MT=8

# Vivotek IP8172
#width=2560
#height=1920
#width2=2560
#height2=1920
#fov=115.3
#integrate=20

# Vivotek IP9171
width=2048
height=1536
width2=2048
height2=1536
fov=115
integrate=20

# Vivotek IP8151
#width=1280
#height=1024
#width2=1280
#height2=1024
#fov=115
#integrate=30

# TANDBERG PrecisionHD
#width=1280
#height=720
#width2=1280
#height2=720
#fov=80
#integrate=20


# GoPro Hero4 Silver
#width=3840
#height=2160
#width2=3840
#height2=2160
#fov=126
#integrate=4

let halfintegrate=$integrate/2

stage=1

astrometry=/etc/astrometry.cfg
if [ "$#" -eq 1 ]; then
    # Useful to restart the process at a certain stage
    stage=$1
fi

# Check prerequisites
[[ $(type -P convert) ]] || { echo '"convert" missing.  Install ImageMagick (apt-get install imagemagick)'; exit 1; }
[[ $(type -P composite) ]] || { echo '"composite" missing.  Install ImageMagick (apt-get install imagemagick)'; exit 1; }
[[ $(type -P nona) ]] || { echo '"nona" missing.  Install the Hugin tools (apt-get install hugin-tools)'; exit 1; }
[[ $(type -P pto_gen) ]] || { echo '"pto_gen" missing.  Install the Hugin tools (apt-get install hugin-tools)'; exit 1; }
[[ $(type -P pto_var) ]] || { echo '"pto_var" missing.  Install the Hugin tools (apt-get install hugin-tools)'; exit 1; }
[[ $(type -P pto_merge) ]] || { echo '"pto_merge" missing.  Install the Hugin tools (apt-get install hugin-tools)'; exit 1; }
[[ $(type -P pano_trafo) ]] || { echo '"pano_trafo" missing.  Install the Hugin tools (apt-get install hugin-tools)'; exit 1; }
[[ $(type -P avconv) ]] || { echo '"avconv" missing.  Install the libav tools (apt-get install libav-tools)'; exit 1; }
[[ $(type -P solve-field) ]] || { echo '"solve-field" missing.  Install astrometry.net (apt-get install astrometry.net)'; exit 1; }
[[ $(type -P parallel) ]] || { echo '"parallel" missing.  Install parallel (apt-get install parallel)'; exit 1; }
[[ $(type -P MP4Box) ]] || { echo '"MP4Box" missing.  Install it (apt-get install gpac)'; exit 1; }
[[ -e /etc/astrometry.cfg ]] || { echo '/etc/astrometry.cfg missing.'; exit 1; }
[[ -e /etc/meteor.cfg ]] || { echo '/etc/meteor.cfg.  Example file:'; echo '[astronomy]'; echo 'latitude = 60.21191'; echo 'longitude = 10.75175'; echo 'elevation = 590.0'; echo 'temperature = 5.0'; echo 'pressure = 980.0'; echo; echo '[station]'; echo 'name = harestua'; echo 'code = HAR'; exit 1; }
[[ -x ~/bin/noisemask ]] || { echo '~/bin/noisemask missing.'; exit 1; }
[[ -x ~/bin/timestamp ]] || { echo '~/bin/timestamp missing.'; exit 1; }
[[ -x ~/bin/avgstack ]] || { echo '~/bin/avgstack missing.'; exit 1; }
[[ -x ~/bin/addstack ]] || { echo '~/bin/addstack missing.'; exit 1; }
[[ -x ~/bin/parsexy ]] || { echo '~/bin/parsexy missing.'; exit 1; }
[[ -x ~/bin/stack ]] || { echo '~/bin/stack missing.'; exit 1; }
[[ -x ~/bin/findstar.py ]] || { echo '~/bin/findstar.py missing.'; exit 1; }
[[ -x ~/bin/map.py ]] || { echo '~/bin/map.py missing.'; exit 1; }
[[ -x ~/bin/altaz.py ]] || { echo '~/bin/altaz.py missing.'; exit 1; }
[[ -x ~/bin/drawgrid.py ]] || { echo '~/bin/drawgrid.py missing.'; exit 1; }
[[ $(type -P python3) ]] || { echo '"python3" missing.'; exit 1; }
if ! $(python3 -c "import ephem" &> /dev/null); then echo 'Ephem missing.  Install it (pip3 install ephem).'; exit 1; fi

alias ffmpeg=avconv

let snap=$width/800

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

    nona -z DEFLATE -o out_$count- -m TIFF_m lens_$count.pto $1 dummy.jpg 2>> calibration.log

    # Try to solve it
    mkdir -p tmp_$count
    rm -f out_$count.corr solutions_$count.txt
    touch solutions_$count.txt
    solve-field -p --scale-low 20 --odds-to-solve 10000000 -c 0.1 -l 120 -m tmp_$count -b $4 --sigma 0 -o out_$count --overwrite out_$count-0000.tif 2>> calibration.log > /dev/null
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

files=$@
step=10

mkdir -p out
if [ $stage -eq 1 ]; then
    echo $files > out/files.txt;
else
    files=$(cat out/files.txt)
fi

for token in $files; do numfiles=$((numfiles+1)); done

for i in $files; do
    if [ ${i: -4} == ".jpg" ]; then
	dest="$dest $(basename ${i%.*})"
    fi
    if [ ${i: -4} == ".mp4" ]; then
	file=$(echo $i | sed 's/[\/_]/ /g;s/\.mp4//' | awk -F'[ ]' '{for(i=NF;i;i--)printf("%s"(i>1?" ":"\n"),$i)}'| awk '{print $2"_"$4$3$1}')
	dest="$dest $file"
    fi
done

cd out
touch calibration.log

case "$stage" in
"1")

export width
export height
export numfiles
export count

function process {
    i=$1
    lockfile -l 5 out/calibrate.lock
    count=$(cat out/count)
    let count=$count+1
    echo $count > out/count
    rm -f out/calibrate.lock
    echo -en "\e[0K\rProcessing file $count/$numfiles"
    if [ ${i: -4} == ".jpg" ]; then
	file=$(basename ${i%.*})
	convert -blur 0x32 $i out/tmp_$file.png
	convert $i out/tmp_$file.png -fx "u-v" -brightness-contrast 45x60 out/outimage_$file.jpg
	timestamp=$((echo -n $(date +%s -u -d "$(echo ${i:0 -18:14} | sed 's/\(....\)\(..\)\(..\)\(..\)\(..\)\(..\)/\1-\2-\3 \4:\5:\6/')");echo +$halfintegrate)|bc)
	echo $timestamp >> out/timestamps.txt
    fi
    if [ ${i: -4} == ".mp4" ]; then
	lockfile -l 5 calibrate_ts.txt
	if [[ ${i:0 -18} =~ 20[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].mp4 ]]; then
	    # Extract timestamp from file if available
	    timestamp=$((echo -n $(date +%s -u -d "$(echo ${i:0 -18:14} | sed 's/\(....\)\(..\)\(..\)\(..\)\(..\)\(..\)/\1-\2-\3 \4:\5:\6/')");echo +$halfintegrate)|bc)
	    file=$(basename $i .mp4)
	else
	    # Otherwise extract from the video
	    timestamp=$((echo -n $(~/bin/timestamp $i | tail -n1);echo +$halfintegrate)|bc)
	    file=$(echo $i | sed 's/[\/_]/ /g;s/\.mp4//' | awk -F'[ ]' '{for(i=NF;i;i--)printf("%s"(i>1?" ":"\n"),$i)}'| awk '{print $2"_"$4$3$1}')
	fi
	echo $timestamp >> out/timestamps.txt
	rm -f calibrate_ts.txt
	ffmpeg -loglevel quiet -i $i -v warning -t $integrate -vsync 0 -nostdin -pix_fmt yuv420p -vf scale=$width:$height -f rawvideo - 2>> out/calibration.log | ~/bin/avgstack $width $height > out/$file.jpg
	convert -blur 0x32 out/$file.jpg out/tmp_$file.png
	#convert -blur 0x8 -brightness-contrast 2x0 out/$file.jpg out/tmp.png
	ffmpeg -loglevel quiet -loop 1 -i out/tmp_$file.png -i $i -v info -t $integrate -vsync 0 -nostdin -pix_fmt yuv420p -f rawvideo -filter_complex "[0:0]setsar=1,format=rgb24[a];[1:0]setsar=1,format=rgb24[b];[b][a]blend=all_mode='subtract',format=yuva422p10le" - 2>> out/calibration.log | ~/bin/addstack $width $height > out/outimage_$file.jpg
    fi
    mask=$(dirname $i)/../../mask.jpg
    if [ -e "$mask" ]; then
	convert out/outimage_$file.jpg out/mask.jpg -fx "u*v" out/enh-$file.png
    else
	convert out/outimage_$file.jpg out/enh-$file.png
    fi
    rm out/outimage_$file.jpg out/tmp_$file.png
}

export -f process
export integrate
export halfintegrate

cd - > /dev/null
echo "1/11 Decode video, reduce noise, extract timestamp, remove background and amplify" | tee -a out/calibration.log
count=0;
mask=$(dirname $i)/../../mask.jpg
if [ -e "$mask" ]; then
    convert -geometry $width"x"$height"!" $mask out/mask.jpg
fi
echo 0 > out/count
rm -f out/calibrate.lock
rm -f out/timestamps.txt
parallel -u -j $MT process {1} ::: $files
sort out/timestamps.txt > out/ts.txt
mv out/ts.txt out/timestamps.txt

#for i in $files; do
#    process $i
#done
cd - > /dev/null

echo

date | tee -a calibration.log
;&
"2")
export numfiles

function static {
    i=$1
    lockfile -l 5 calibrate.lock
    count=$(cat count)
    let count=$count+1
    echo $count > count
    rm -f calibrate.lock
    echo -en "\e[0K\rProcessing file $count/$numfiles"
    if [ ${i: -4} == ".jpg" ]; then
	file=$(basename ${i%.*})
    else
	file=$(echo $i | sed 's/[\/_]/ /g;s/\.mp4//' | awk -F'[ ]' '{for(i=NF;i;i--)printf("%s"(i>1?" ":"\n"),$i)}'| awk '{print $2"_"$4$3$1}')
    fi
    convert enh-$file.png static.png -fx "(u-$sub/256)*v*2" -morphology Thinning 3:0,0,0,0,1,0,0,0,0 -colorspace gray $file.png
}

export -f static

echo "2/11 Remove static background" | tee -a calibration.log
count=0;
(for i in enh-full*.png; do convert $i x.yuv; cat x.yuv; done) | ~/bin/noisemask $width $height 2> sub.txt | convert -depth 8 -size $width"x"$height yuv:- static.png
sub=$(cat sub.txt)
export sub

echo 0 > count
rm -f calibrate.lock
parallel -u -j $MT static {1} ::: $files

#for i in $files; do
#    static $i
#done

echo

date | tee -a calibration.log
;&
"3")

echo "3/11 Find candidate stars" | tee -a calibration.log
count=0
for i in full*[0-9].png; do
    let count=$count+1
    echo -en "\e[0K\rProcessing file $count/$numfiles"
    solve-field --sigma 1 --overwrite -l 0.1 $i 2>> calibration.log > /dev/null
    rm -f /tmp/tmp.fits*
done
echo

date | tee -a calibration.log
;&
"4")
echo "4/11 Analyse candidates, look for things moving and record paths" | tee -a calibration.log
~/bin/parsexy 9 full*axy > stars.txt

# Add timestamps as the first column
mv stars.txt stars2.txt
paste timestamps.txt stars2.txt | head -n$(wc -l stars2.txt|awk '{print $1}') > stars.txt

date | tee -a calibration.log
;&
"5")
echo "5/11 Create masks for moving objects" | tee -a calibration.log
# Also create composite image (stars.jpg)
sed 's/\t-------,-------//g;s/\t/ -draw point\\ /g' < stars2.txt | awk -v a="$dest" -v num=2 '{split(a, A, / /); printf("convert -size '$width'x'$height' xc:black -fill white "$0" -morphology Dilate Disk:5 -blur 0x2 star_" A[num++] ".jpg\n");}' | bash
(sed 's/\t-------,-------//g;s/\t/ -draw point\\ /g' < stars2.txt | awk '{printf("convert -size '$width'x'$height' xc:black -fill white "$0" -morphology Dilate Plus:2 -depth 8 yuv:\n");}' | bash) | ~/bin/stack $width $height > stars.jpg
rename -f s/star_full/star/ star_full*

date | tee -a calibration.log
;&
"6")
echo "6/11 Create images with everything masked out except the detected stars" | tee -a calibration.log
line=0;
for i in $dest; do
    let line=$line+1
    echo -en "\e[0K\rProcessing file $line/$numfiles"
    echo convert $i.png $(echo $i|sed "s/full/star/").jpg -fx "u*v" -fill white $(sed $line'q;d' < stars2.txt | sed 's/\t-------,-------//g;s/\t/ -draw point\\ /g') $(echo $i|sed "s/full/clean/").jpg | bash
    #echo convert $i $(echo $i|sed "s/full/star/;s/png/jpg/") -fx "u*v" $(echo $i|sed "s/full/clean/;s/png/jpg/") | bash
done

echo

count=0
date | tee -a calibration.log
;&
"7")
echo "7/11 Assign identity numbers" | tee -a calibration.log
for i in clean*.jpg; do
    let count=count+1
    echo -en "\e[0K\rProcessing file $count/$numfiles"
    echo $(cat stars2.txt | sed $count'q;d' | sed 's/\t\([0-9\.]*\),\([0-9\.]*\)/ -annotate +\1+\2\n/g;s/-------,-------/ -annotate -100-100\n/g' | head -n -1 | awk '{print $0" "++i}') | awk '{printf("convert '$i' -pointsize 16 -fill green "$0" '$(echo $i | sed s/clean/key/)'\n");}' | bash
done
echo

date | tee -a calibration.log
;&
"8")
echo "8/11 Plate solving" | tee -a calibration.log
line=0
rm -f stars.tmp.id

# Create dummy frame where x=az y=90-alt
convert -size 360x180 xc:darkblue dummy.jpg

# Create initial Hugin project file
pto_gen -f $fov $(ls -1 clean*.jpg|head -n1) -p 3 -o /dev/stdout 2>> calibration.log | sed 's/ r:CROP//;s/f2 w[0-9]* h[0-9]* v/f2 w36000 h18000 v/' > lens.pto
pto_gen -f 360 $(ls -1 clean*.jpg|head -n1) -p 4 dummy.jpg -o /dev/stdout 2>> calibration.log | sed 's/ r:CROP//;s/f2 w[0-9]* h[0-9]* v/f2 w36000 h18000 v/' > dummy.pto
pto_merge lens.pto dummy.pto -o lens.pto 2>> calibration.log > /dev/null
pto_var --opt r0,p0,y0,a0,b0,c0,d0,e0,v0 -o lens.pto lens.pto 2>> calibration.log > /dev/null

# Thoby fisheye
#sed 's/ f3 / f20 /' lens.pto > lens3.pto; mv lens3.pto lens.pto

for clean in clean*.jpg; do
    if [ $(($line % 8)) == 0 ]; then
	minyaw=-55
	maxyaw=55
	minpitch=-40
	maxpitch=40
    fi
    let line=$line+1
    if [ $(($line % 2)) == 0 ]; then continue; fi

    echo -en "\e[0K\rProcessing file $line/$numfiles"

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
    parallel -u -j $(echo $(parallel --number-of-cores)/2 | bc) solve $clean {1} {2} $astrometry ::: $yaws ::: $pitches

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

    # Associate the solutions with the corresponding column in the stars.txt file
    star=0
    for i in $(sed $line'q;d' stars.txt); do
	if [ -s $clean.solutions ]; then
	    x=$(echo $i | sed s/,.*//)
	    y=$(echo $i | sed s/.*,//)
	    # Is the pixel close enough?
	    if [[ $x != *--* ]]; then
		awk '{if (('$x'-$1)*('$x'-$1)+('$y'-$2)*('$y'-$2) < '$snap'*'$snap') {$1=$2=""; print "'$star': "$0}}' $clean.solutions
		awk '{if (('$x'-$1)*('$x'-$1)+('$y'-$2)*('$y'-$2) < '$snap'*'$snap') {$1=$2=""; print "'$star' "$0}}' $clean.solutions >> stars.tmp.id
	    fi
	    let star=$star+1
	fi
    done
done

# Collapse identical solutions, eliminate sporadic (and likely wrong) solutions, and pick the most frequent of conflicting solutions
sort stars.tmp.id|uniq -c|awk '{print $2" "$1" "$3" "$4" "$5" "$6" "$7}' |sort -nrk2|sort -ns|awk '{c=c+1; for(i=c;i<$1;i++) {print i} if (c<=$1) {if ($2 > 5) {print $1" "$3" "$4" "$5" "$6" "$7} else {$1--}};c=$1}' > stars.id

if [ ! -s stars.id ]; then
    echo No solution.
    exit 1
fi

pano_modify --canvas=36000x18000 --fov=360x180 --projection=2 lens.pto -o lens.pto > /dev/null 2>> calibration.log

cp lens.pto lens.pto.bak
cp stars.id stars.id.bak

date | tee -a calibration.log
;&
"9")
echo "9/11 Solve lens" | tee -a calibration.log
cp lens.pto.bak lens.pto

~/bin/map.py stars.txt stars.id | awk '{printf("c n0 N1 x%s y%s X%s Y%s t0\n", $1, $2, $3*100, (90-$4)*100)}' > stars.lens
cat stars.lens >> lens.pto

echo Coverage:
convert -blur 8x8 -auto-gamma -negate -geometry 80 stars.jpg pbm:- | pbmtoascii - | sed 's/\S/X/g'

cat stars.id
cpclean -n 1 lens.pto -o lens.pto 2>&1 | grep Removed
error=$(autooptimiser -n lens.pto -o lens2.pto 2>&1 | grep units | tail -n1)

echo $error | awk '{printf("Average error: %.3f arc minutes\n", $4*60/100)}'

poor=$(echo $error | awk '{ print($4*60/100 > 2) }')

if [ "$poor" -eq 1 ]; then
    awk '{printf($0); if ($2 != "" && $3 != "") { system("~/bin/findstar.py -m "$2" "$3) } else { printf("\n"); }}' < stars.id > stars2.id
fi

mag=$(awk '{printf("%.1f\n", $5-0.1)}' stars2.id|sort -n|tail -1)
while [ "$poor" -eq 1 ] && [ $(echo "$mag > 1.2" | bc -l) -eq "1" ]; do
    echo Poor solution.  Improve by eliminating the weakest stars "("$mag")".

    awk '{if ($NF < '$mag') {$NF=""; print $0} else {print $1}}' < stars2.id > stars3.id
    
    ~/bin/map.py stars.txt stars3.id | awk '{printf("c n0 N1 x%s y%s X%s Y%s t0\n", $1, $2, $3*100, (90-$4)*100)}' > stars.lens
    cp lens.pto.bak lens.pto
    cat stars.lens >> lens.pto

    cpclean -n 1 lens.pto -o lens.pto 2>&1 | grep Removed
    error=$(autooptimiser -n lens.pto -o lens2.pto 2>&1 | grep units | tail -n1)

    echo $error | awk '{printf("Average error: %.3f arc minutes\n", $4*60/100)}'
    poor=$(echo $error | awk '{ print($4*60/100 > 2) }')

    mag=$(bc <<< $mag-0.1)
done

if [ "$poor" -eq 1 ]; then
    echo Still poor solution.  Improve by eliminating random stars.
    
    num=$(wc -l < stars.lens)
    len=$(wc -l < lens2.pto)
    x=2
    for i in $(seq 1 100 $num); do

	~/bin/map.py stars.txt stars3.id | awk '{printf("c n0 N1 x%s y%s X%s Y%s t0\n", $1, $2, $3*100, (90-$4)*100)}' | awk 'NR % '$x' == 0' > stars.lens
	let x=x+1
	cp lens.pto.bak lens.pto
	cat stars.lens >> lens.pto

	cpclean -n 1 lens.pto -o lens.pto 2>&1 | grep Removed
	error=$(autooptimiser -n lens.pto -o lens2.pto 2>&1 | grep units | tail -n1)


	cpclean -n 1 lens2.pto -o lens2.pto 2> /dev/null >> calibration.log
	error=$(autooptimiser -n lens2.pto -o lens2.pto 2> /dev/null | grep units | tail -n1)
	dist=$(grep f3 lens2.pto | sed 's/.* a//;s/ d.*//;s/[bc]//g'|awk 'function abs(x){return ((x < 0.0) ? -x : x)} {print abs($1)+abs($2)+abs($3)}')
	lastlen=$len
	len=$(wc -l < lens2.pto)
	echo "Lens distortion: "$dist
	if [ $(echo $dist | awk '{ print($1 < 2)}') -eq 1 ] || [ $len -eq $lastlen ] ; then
	    break
	fi
    done

    echo $error | awk '{printf("Average error: %.3f arc minutes\n", $4*60/100)}'
fi

~/bin/drawgrid.py lens2.pto

count=0
date | tee -a calibration.log
;&
"10")

export numfiles

function label {
    i=$1
    lockfile -l 5 calibrate.lock
    count=$(cat count)
    let count=$count+1
    echo $count > count
    rm -f calibrate.lock

    echo -en "\e[0K\rProcessing file $count/$numfiles"
    if [ ${i: -4} == ".jpg" ]; then
	file=$(basename ${i%.*})
    else
	file=$(echo $i | sed 's/[\/_]/ /g;s/\.mp4//' | awk -F'[ ]' '{for(i=NF;i;i--)printf("%s"(i>1?" ":"\n"),$i)}'| awk '{print $2"_"$4$3$1}')
    fi
    convert $(echo $file | sed "s/full/clean/").jpg -alpha set -channel alpha -fx "(r+g+b)/3" png:- | convert $file.jpg - -composite tmp_$file.png
    composite -blend 80 tmp_$file.png grid.png tmp2_$file.png
    ts=$(sed $count'q;d' timestamps.txt)
    awk '{if (!$2) {print(""); } else { system("~/bin/altaz.py '$ts' "$2" "$3)}}' stars.id > pos_$file.txt
    echo $(cat stars2.txt | sed $count'q;d' | sed 's/\t\([0-9\.]*\),\([0-9\.]*\)/ -annotate +\1+\2\n/g;s/-------,-------/ -annotate -100-100\n/g' | head -n -1 | awk '{l=$0; "grep ^"++i" stars.id" | getline; ra=$2; dec=$3; $1=$2=$3=""; print l" \""$0" ";"sed "i"q\\;d pos_'$file'.txt" | getline; if ($1 >= 0) {printf("[%6.2f%6.2f]",$1,$2)}; print "\""}') | awk '{printf("convert tmp2_'$file'.png -pointsize 16 -fill green "$0" '$(echo $file | sed s/full/mixed/).jpg'\n");}' | bash
    #rm pos_$file.txt tmp_$file.png tmp2_$file.png
}

export -f label
echo 0 > count

echo "10/11 Add labels in original frames" | tee -a calibration.log
for i in $files; do label $i; done
#parallel -u label {1} ::: $files
echo

date | tee -a calibration.log
;&
"11")
echo "11/11 Create videos" | tee -a calibration.log
H264="-profile:v baseline -level:v 1.0"
echo -en "\e[0K\rProcessing video 1/8"
cat full*.jpg | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec mjpeg -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Original frames':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" orig.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 2/8"
cat enh-full*.png | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec png -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Background removal and amplification':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" enhanced.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 3/8"
cat full*[0-9].png | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec png -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Static noise removal':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" enhanced2.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 4/8"
cat full*-objs.png | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec png -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Star candidates':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" cand.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 5/8"
cat star_*.jpg | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec mjpeg -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Motion detection':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" stars.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 6/8"
cat key_*.jpg | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec mjpeg -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Labels':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" key.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 7/8"
cat mixed*.jpg | ffmpeg -nostdin -y -vsync 0 -f image2pipe -r 10 -vcodec mjpeg -i - -pix_fmt yuv420p $H264 -vcodec libx264 -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf:text='Amplified detected stars in original frames':fontcolor=white:x=20:y=20:fontsize=36,crop=w=$width2:h=$height2:x=0:y=0" mixed.mp4 >> calibration.log 2>&1
echo -en "\e[0K\rProcessing video 8/8"
rm -f combined.mp4; MP4Box -fps 10 -add orig.mp4 -cat enhanced.mp4 -cat enhanced2.mp4 -cat cand.mp4 -cat stars.mp4 -cat key.mp4 -cat mixed.mp4 combined.mp4 >> calibration.log 2>&1
echo
echo "Done!"
echo "Full transcript in calibration.log"
date | tee -a calibration.log
esac
