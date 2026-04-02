#!/bin/bash

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root"
    exit 1
fi

echo Detecting cameras...
hosts=$(arp-scan --interface=eth0 --localnet|grep Vivotek|cut -f1);

count=1
for i in $hosts; do
    cam=$(curl -su root:XXCAMPASSWORDXX "http://"$i"/cgi-bin/admin/getparam.cgi?system_info_modelname" | awk -F\' '{ print $2 }')
    echo $count\) $i \($cam\)
    let count=count+1
done

correct=no

until [[ "${correct:0:1}"  == "y" ]]; do

    read -p "Which cameras should we use (give a space separated list, e.g.: 1 2)? " cams

    arr=($hosts)
    count=1
    for i in $cams; do
	usehosts="$usehosts ${arr[i-1]}"
	let count=count+1
    done

    echo You selected: $usehosts
    read -p "Is the above correct? > " correct
    echo
done

echo $usehosts > /etc/cameras.ip.new
echo $cams > /etc/cameras.id.new
mv /etc/cameras.ip.new /etc/cameras.ip
mv /etc/cameras.id.new /etc/cameras.id
