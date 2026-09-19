#!/bin/bash

# Deletes the oldest 8-digit date directory under $1 when its filesystem is
# >=95% full.  $1 must resolve to a real directory under the resolved
# /meteor root, so a planted link cannot redirect the rm -rf elsewhere.
# (/meteor itself may legitimately be a symlink to mounted storage, so we
# compare resolved paths rather than refuse links outright.)

if [ -z "$1" ]; then
    exit 1
fi

root=$(realpath -e -- /meteor 2>/dev/null) || exit 1
dir=$(realpath -e -- "$1" 2>/dev/null) || exit 1
case "$dir" in
    "$root"/*) ;;
    *) echo "Refusing to operate outside $root: $1" >&2; exit 1 ;;
esac

if [ ! -d "$dir" ]; then
    exit 1
fi

if [ $(df -k --output=pcent "$dir" | tail -n1 | sed 's/%//') -ge 95 ]; then

    min_dirs=3

    [[ $(find "$dir" -maxdepth 1 -type d | wc -l) -ge $min_dirs ]] &&
        IFS= read -r -d $'\0' line < <(find "$dir" -maxdepth 1 -type d \( -name "20[0-9][0-9][0-9][0-9][0-9][0-9]" \) -printf '%T@ %p\0' 2>/dev/null | sort -z -n)
    file="${line#* }"

    if [ ! -z "$file" ]; then
        # Containment re-check on the resolved path right before deletion.
        resolved=$(realpath -e -- "$file" 2>/dev/null) || exit 1
        case "$resolved" in
            "$dir"/*) ;;
            *) echo "Refusing to remove $file (resolves to $resolved)" >&2; exit 1 ;;
        esac
        rm -rf -- "$resolved"
    fi
fi
