#!/bin/bash

cd $(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"/../../meteor"
# Fetch remote event feed. Security notes:
# - Do NOT use curl -k: certificate validation is the only thing standing
#   between this cron job and a man-in-the-middle injecting shell commands.
# - Never pass remote fields through system(): validate $13/$14 strictly and
#   write the record with awk's print redirection instead of echo + system.
curl -s --connect-timeout 15 --max-time 60 https://www.astro.uu.se/~meteor/latest.php |
  grep -v ALISA | grep -v MRK | grep -v "^#" |
  awk '$13 ~ /^[a-zA-Z0-9_-]+$/ && $14 ~ /^[0-9]+$/ {
      dir  = sprintf("%s/%s/cam1", strftime("%Y%m%d/%H%M%S", $14, 1), $13);
      file = sprintf("%s/%s-%s.txt", dir, $13, strftime("%Y%m%d%H%M%S", $14, 1));
      system("mkdir -p " dir);
      print $0 > file;
      close(file);
  }'

#rm -rf $(find . -mindepth 2 -maxdepth 2 -type d -mtime +1 '!' -exec test -e "{}/index.php" ';' -print | egrep ".*[0-9][0-9][0-9][0-9][0-9][0-9]")
