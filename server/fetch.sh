#!/bin/bash

STATION="$1"
PORT="$2"
DIR="$3"

# Use testing path if available, otherwise use server path
TEST_DIR="/home/steinar/norskmeteornettverk.no/nmn/server"
SERVER_DIR="/home/httpd/norskmeteornettverk.no/nmn/server"

if [ -d "$TEST_DIR" ]; then
    SCRIPT_DIR="$TEST_DIR"
else
    SCRIPT_DIR="$SERVER_DIR"
fi

python3 "$SCRIPT_DIR/fetch.py" "$STATION" "$PORT" "$DIR"
