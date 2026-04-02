#!/bin/bash

cd ~/nmn

do_if_newer() {
    src=$1
    bin=$2
    shift 2
    if [[ ! -f $bin || $src -nt $bin ]]; then "$@"; fi
}

do_if_newer src/metdetect.c bin/metdetect g++ src/metdetect.c -o bin/metdetect -I$HOME/src/hugin -O6 -ljpeg -lm -lavutil -lavcodec -pthread -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0
do_if_newer src/parsexy.c bin/parsexy gcc src/parsexy.c -O3 -o bin/parsexy -lm
