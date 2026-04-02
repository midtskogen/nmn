#!/bin/bash

dark=$(/home/meteor/bin/sunriset $(grep latitude /etc/meteor.cfg | sed s/.*=//) $(grep longitude /etc/meteor.cfg | sed s/.*=//g) -10)

for i in $(cat /etc/cameras.ip); do
    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')
    if [[ "$cam" != "IP8172" ]]; then
	continue
    fi
    echo Configuring $i:
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?ircutcontrol_mode=schedule"
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?ircutcontrol_daymodebegintime="$(echo $dark | cut -d" " -f2)
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?ircutcontrol_daymodeendtime="$(echo $dark | cut -d" " -f1)
done
