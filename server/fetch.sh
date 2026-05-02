#!/bin/bash

STATION="$1"
PORT="$2"
DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/fetch.py" "$STATION" "$PORT" "$DIR"
