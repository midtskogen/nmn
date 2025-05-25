#!/usr/bin/env python3

# Telnet root password: xmhdipc

from dvrip import DVRIPCam
from pprint import pprint

#ips = [ "192.168.1.10" ]
ips = [ "192.168.76.71" ]

for ip in ips:
    cam = DVRIPCam(ip, user='admin', password='')
    cam.login()
    cam.set_time()

    ip_settings = cam.get_info("NetWork.NetCommon")
    # 0x47 = 71
    # 0x48 = 72
    # 0x49 = 73
    # 0x4a = 74
    # 0x4b = 75
    # 0x4c = 76
    # 0x4d = 77
#    ip_settings["HostIP"] =  '0x4B4CA8C0'
#    ip_settings["GateWay"] = '0x014CA8C0'
    ip_settings["HostIP"] =  '0x6600000A'
    ip_settings["GateWay"] = '0x8A00000A'
    cam.set_info("NetWork.NetCommon", ip_settings)
    print(ip_settings)

# Disconnect
cam.close()
