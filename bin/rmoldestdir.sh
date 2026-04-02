#!/bin/bash

if [ -e "$1" ] && [ $(df -k --output=pcent "$1" | tail -n1 | sed 's/%//') -ge 95 ]; then

    dir="$1"
    min_dirs=3

    [[ $(find "$dir" -maxdepth 1 -type d | wc -l) -ge $min_dirs ]] &&
        IFS= read -r -d $'\0' line < <(find "$dir" -maxdepth 1 -type d \( -name "20[0-9][0-9][0-9][0-9][0-9][0-9]" \) -printf '%T@ %p\0' 2>/dev/null | sort -z -n)
    file="${line#* }"

    if [ ! -z "$file" ]; then
        rm -rf "$file"
    fi
fi
