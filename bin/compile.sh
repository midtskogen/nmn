#!/bin/bash

cd ~/nmn

do_if_newer() {
    src=$1
    bin=$2
    shift 2
    if [[ ! -f $bin || $src -nt $bin ]]; then "$@"; fi
}

do_if_newer src/addstack.c bin/addstack gcc src/addstack.c -O3 -o bin/addstack -ljpeg
do_if_newer src/avgstack.c bin/avgstack gcc src/avgstack.c -O3 -o bin/avgstack -ljpeg
do_if_newer src/metdetect.c bin/metdetect g++ src/metdetect.c -o bin/metdetect -I$HOME/src/hugin -O6 -ljpeg -lm -lavutil -lavcodec -pthread -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0
do_if_newer src/noisemask.c bin/noisemask gcc src/noisemask.c -O3 -o bin/noisemask
do_if_newer src/parsexy.c bin/parsexy gcc src/parsexy.c -O3 -o bin/parsexy -lm
do_if_newer src/stack.c bin/stack gcc src/stack.c -msse4.2 -O3 -o bin/stack -ljpeg -lavif
do_if_newer src/sunriset.c bin/sunriset gcc src/sunriset.c -O3 -o bin/sunriset -lm
do_if_newer src/timestamp.c bin/timestamp gcc src/timestamp.c -O3 -o bin/timestamp
