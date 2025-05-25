#!/usr/bin/env python3

import sys
import ipaddress
import argparse
from dvrip import DVRIPCam
from pprint import pprint

def ip_to_hex(ip: str) -> str:
    """Convert dotted IP to little endian 32 bit hex string."""
    return '0x' + ''.join([f'{int(octet):02X}' for octet in reversed(ip.split('.'))])

def main():
    parser = argparse.ArgumentParser(description="Change IP address of DVRIP camera")
    parser.add_argument("new_ip", help="New IP address with optional /mask (e.g. 192.168.76.75/24)")
    parser.add_argument("original_ip", nargs='?', default="192.168.1.10",
                        help="Current IP of the camera (default: 192.168.1.10)")
    parser.add_argument("-p", "--password", default="", help="Password for the camera (default: empty)")
    args = parser.parse_args()

    original_ip = args.original_ip
    new_ip_input = args.new_ip
    password = args.password

    # Handle optional CIDR
    if '/' not in new_ip_input:
        new_ip_input += '/24'

    try:
        ip_interface = ipaddress.IPv4Interface(new_ip_input)
    except ValueError as e:
        print(f"Invalid IP input: {e}")
        sys.exit(1)

    new_ip = str(ip_interface.ip)
    net = ip_interface.network
    gateway_ip = str(list(net.hosts())[0])  # First usable address

    print(f"Connecting to camera at {original_ip}")
    cam = DVRIPCam(original_ip, user='admin', password=password)

    if not cam.login():
        print("Login failed.")
        sys.exit(1)

    cam.set_time()

    ip_settings = cam.get_info("NetWork.NetCommon")
    ip_settings["HostIP"] = ip_to_hex(new_ip)
    ip_settings["GateWay"] = ip_to_hex(gateway_ip)

    cam.set_info("NetWork.NetCommon", ip_settings)
    pprint(ip_settings)

    cam.close()
    print("Camera IP update complete.")

if __name__ == '__main__':
    main()
