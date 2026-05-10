#!/bin/bash

cd $(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"/../../meteor"
curl -ks --connect-timeout 15 --max-time 60 https://www.astro.uu.se/~meteor/latest.php |grep -v ALISA |grep -v MRK|grep -v "^#"|awk '{dir=sprintf("%s/%s/cam1", strftime("%Y%m%d/%H%M%S", $14, 1), $13); file=sprintf("%s/%s-%s.txt", dir, $13, strftime("%Y%m%d%H%M%S", $14, 1)); cmd=sprintf("mkdir -p %s; echo \"%s\" > %s", dir, $0, file); system(cmd)}'

#rm -rf $(find . -mindepth 2 -maxdepth 2 -type d -mtime +1 '!' -exec test -e "{}/index.php" ';' -print | egrep ".*[0-9][0-9][0-9][0-9][0-9][0-9]")
