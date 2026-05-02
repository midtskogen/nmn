#!/bin/bash

ROOT=/home/httpd/norskmeteornettverk.no/cam

for s in kristiansand hofsoy orsta moss loten finnskogen; do
    (
    for i in $ROOT/$s/cam*; do
        for ext in jpg avif; do
            src_files="/cam${i#$(dirname $(dirname "$i"))}/snapshot.$ext"
	    for src_file in $src_files; do
		if [ -f "$src_file" ]; then
                    cp -a "$src_file" $i
                    if [ "$ext" != "jpg" ]; then
			convert "/cam${i#$(dirname $(dirname "$i"))}/snapshot.$ext" $i/snapshot.jpg
                    fi
		fi
	    done
        done
    done
    ) &
done
wait
