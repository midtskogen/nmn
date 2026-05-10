#!/bin/bash

STATION="$1"
PORT="$2"
DIR="$3"

LOG_FILE="/tmp/fetch_sh.log"

echo "$(date): fetch.sh started with args: $STATION $PORT $DIR" >> "$LOG_FILE"
echo "$(date): Running as user: $(whoami), pwd: $(pwd)" >> "$LOG_FILE"

# Use testing path if available, otherwise use server path
TEST_DIR="/home/steinar/norskmeteornettverk.no/nmn/server"
SERVER_DIR="/home/httpd/norskmeteornettverk.no/nmn/server"

# Redirect stderr when checking test dir to avoid permission error messages
if test -d "$TEST_DIR" 2>/dev/null; then
    SCRIPT_DIR="$TEST_DIR"
    echo "$(date): Using TEST_DIR: $TEST_DIR" >> "$LOG_FILE"
else
    SCRIPT_DIR="$SERVER_DIR"
    echo "$(date): Using SERVER_DIR: $SERVER_DIR (test dir check failed or not found)" >> "$LOG_FILE"
fi

echo "$(date): SCRIPT_DIR=$SCRIPT_DIR, exists=$(test -d "$SCRIPT_DIR" 2>/dev/null && echo yes || echo no)" >> "$LOG_FILE"
echo "$(date): fetch.py exists=$(test -f "$SCRIPT_DIR/fetch.py" 2>/dev/null && echo yes || echo no)" >> "$LOG_FILE"

python3 "$SCRIPT_DIR/fetch.py" "$STATION" "$PORT" "$DIR" 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "$(date): fetch.sh finished with exit code: $EXIT_CODE" >> "$LOG_FILE"
