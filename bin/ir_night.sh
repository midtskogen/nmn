#!/bin/bash

cameras=$(cat /etc/cameras.ip)

for i in $cameras; do
    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')
    if [[ "$cam" != "IP8172" ]]; then
	continue
    fi
    echo Configuring $i:
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?ircutcontrol_mode=night"
done

for i in $cameras; do
    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')
    if [[ "$cam" != "IP8172" ]]; then
	continue
    fi
    echo Configuring $i:
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoin_c0_rgain=25"
done
