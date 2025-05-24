#!/bin/bash                                                                                                                                          

dark=$(/home/meteor/bin/sunriset $(grep latitude /etc/meteor.cfg | sed s/.*=//) $(grep longitude /etc/meteor.cfg | sed s/.*=//g) -10)
errors=0

for i in $(cat /etc/cameras.ip); do
    if ! ping -c 1 $i &> /dev/null; then
        let errors=errors+1
        continue;
    fi
    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')
    if [[ "$cam" == "IP9171" ]]; then
        continue
    fi
    echo Configuring $i:
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoin_irismode=outdoor"
    let errors=errors+$?
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoin_c0_irismode=outdoor"
    let errors=errors+$?
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoinpreview_irismode=outdoor"
    let errors=errors+$?
done
exit $errors
