#!/bin/bash

# Compile addstack avgstack metdetect noisemask parsexy stack sunriset timestamp
cd ~/nmn
gcc src/addstack.c -O3 -o bin/addstack -ljpeg
gcc src/avgstack.c -O3 -o bin/avgstack -ljpeg
g++ src/metdetect.c -o bin/metdetect -I$HOME/src/hugin -O6 -ljpeg -lm -lavutil -lavcodec -pthread -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0
gcc src/noisemask.c -O3 -o bin/noisemask
gcc src/parsexy.c -O3 -o bin/parsexy -lm
gcc src/stack.c -msse4.2 -O3 -o bin/stack -ljpeg -lavif
gcc src/sunriset.c -O3 -o bin/sunriset -lm
gcc src/timestamp.c -O3 -o bin/timestamp
