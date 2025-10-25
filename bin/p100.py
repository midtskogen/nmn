#!/usr/bin/env python3

"""
tapo.py

A command-line utility for controlling Tapo P100 smart plugs.

This script allows you to turn the device on or off, check its status,
and retrieve detailed device information.

Credentials (IP, username, password) can be provided via command-line
arguments or environment variables.

Environment Variables:
  TAPO_ADDRESS    - The IP address of the Tapo device.
  TAPO_USERNAME   - The email/username for your Tapo account.
  TAPO_PASSWORD   - The password for your Tapo account.

Command-line Arguments:
  --ip IP_ADDRESS  - The IP address of the Tapo device.
  --user USERNAME  - Your Tapo account username (email).
  --pass PASSWORD  - Your Tapo account password.

Commands:
  on              - Turn the device on.
  off             - Turn the device off.
  status          - Check if the device is currently on or off.
  info            - Display detailed device information and usage statistics.
"""

import asyncio
import os
import sys
import argparse
import json
from tapo import ApiClient

async def main():
    """
    Main asynchronous function to parse arguments and control the Tapo device.
    """
    
    # --- Argument Parsing ---
    
    # Define a custom formatter to show help on error
    class HelpOnErrorParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f'Error: {message}\n\n')
            self.print_help()
            sys.exit(2)

    parser = HelpOnErrorParser(
        description='Control a Tapo P100 smart plug from the command line.',
        epilog='Example: tapo.py --ip 192.168.1.20 --user me@example.com --pass mypass on'
    )
    
    parser.add_argument(
        '--ip',
        metavar='IP_ADDRESS',
        type=str,
        help='The IP address of the Tapo P100 device.'
    )
    parser.add_argument(
        '--user',
        '--username',
        dest='user',
        metavar='USERNAME',
        type=str,
        help='The username (email) for your Tapo account.'
    )
    parser.add_argument(
        '--password',
        '--pass',
        '--pw',
        dest='password',
        metavar='PASSWORD',
        type=str,
        help='The password for your Tapo account.'
    )
    
    parser.add_argument(
        'command',
        choices=['on', 'off', 'status', 'info'],
        help='The action to perform: on, off, status, or info.'
    )
    
    args = parser.parse_args()
    
    # --- Get Credentials ---
    
    # Prioritize command-line arguments, then fall back to environment variables
    ip_address = args.ip or os.getenv("TAPO_ADDRESS")
    tapo_username = args.user or os.getenv("TAPO_USERNAME")
    tapo_password = args.password or os.getenv("TAPO_PASSWORD")
    
    # --- Validate Credentials ---
    
    missing_vars = []
    if not ip_address:
        missing_vars.append("IP address (set with --ip or TAPO_ADDRESS)")
    if not tapo_username:
        missing_vars.append("Username (set with --user or TAPO_USERNAME)")
    if not tapo_password:
        missing_vars.append("Password (set with --password or TAPO_PASSWORD)")
        
    if missing_vars:
        print("Error: Missing required configuration.\n", file=sys.stderr)
        for var in missing_vars:
            print(f"- {var}", file=sys.stderr)
        print("\nUse --help for more information.", file=sys.stderr)
        sys.exit(1)

    # --- Execute Command ---
    
    try:
        # Initialize the API client
        client = ApiClient(tapo_username, tapo_password)
        
        # Connect to the P100 device
        device = await client.p100(ip_address)
        
        # --- Command Handling ---
        
        if args.command == 'on':
            print(f"Turning device on at {ip_address}...")
            await device.on()
            print("Device is now on.")
            
        elif args.command == 'off':
            print(f"Turning device off at {ip_address}...")
            await device.off()
            print("Device is now off.")
            
        elif args.command == 'status':
            device_info = await device.get_device_info()
            state = "On" if device_info.device_on else "Off"
            print(state)
            
        elif args.command == 'info':
            print(f"Retrieving information from {ip_address}...")
            device_info_raw = await device.get_device_info()
            device_usage_raw = await device.get_device_usage()
            
            device_info = device_info_raw.to_dict()
            device_usage = device_usage_raw.to_dict()
            
            print("\n--- Device Info ---")
            info_map = {
                'nickname':     'Nickname',
                'model':        'Model',
                'type':         'Type',
                'ip':           'IP Address',
                'mac':          'MAC Address',
                'device_on':    'Current State',
                'on_time':      'On Time (seconds)',
                'rssi':         'RSSI (Signal)',
                'signal_level': 'Signal Level (1-3)',
                'fw_ver':       'Firmware Version',
                'hw_ver':       'Hardware Version',
                'ssid':         'Connected SSID',
                'region':       'Region',
            }
            
            for key, label in info_map.items():
                if key in device_info:
                    value = device_info[key]
                    if key == 'device_on':
                        value = "On" if value else "Off"
                    print(f"{label + ':':<19} {value}")

            print("\n--- Device Usage ---")
            if 'time_usage' in device_usage:
                usage_map = {
                    'today':    'Today',
                    'past7':    'Past 7 Days',
                    'past30':   'Past 30 Days',
                }
                for key, label in usage_map.items():
                    if key in device_usage['time_usage']:
                        # Usage is reported in minutes
                        minutes = device_usage['time_usage'][key]
                        hours = minutes // 60
                        mins = minutes % 60
                        print(f"{label + ':':<19} {minutes} minutes ({hours}h {mins}m)")
            else:
                print("No usage data available.")
                
            # Print any other info not in our map
            print("\n--- Full Device Data (JSON) ---")
            device_info['usage'] = device_usage
            print(json.dumps(device_info, indent=2))

            
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        print("Please check IP address, credentials, and network connection.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Set logging level for the tapo library to avoid excessive debug output
    # You can change 'WARNING' to 'DEBUG' for more verbose logs
    import logging
    logging.getLogger('tapo').setLevel(logging.WARNING)
    
    asyncio.run(main())

