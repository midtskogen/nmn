#!/bin/bash

ROOT=/home/httpd/norskmeteornettverk.no/cam
for s in sorreisa trondheim larvik gran voksenlia harestua skibotn gaustatoppen eiscat tromso lyngseidet hagar; do
    for i in $ROOT/$s/cam*; do
	cp -a /cam${i#$(dirname $(dirname $i))}/snapshot.jpg $i &
    done
done
wait
