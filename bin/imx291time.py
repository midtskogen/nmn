#!/usr/bin/env python3

from dvrip import DVRIPCam

ips = [ "192.168.76.71", "192.168.76.72", "192.168.76.73", "192.168.76.74", "192.168.76.75", "192.168.76.76", "192.168.76.77" ]

for ip in ips:
    try:
        cam = DVRIPCam(ip, user='admin', password='')
        cam.login()
        cam.set_time()
    except:
        pass
