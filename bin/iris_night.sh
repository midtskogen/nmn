#!/bin/bash                                                                                                                                          
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
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoin_irismode=fixed"
    let errors=errors+$?
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoin_c0_irismode=fixed"
    let errors=errors+$?
    curl -u root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/setparam.cgi?videoinpreview_irismode=fixed"
    let errors=errors+$?
done
exit $errors
