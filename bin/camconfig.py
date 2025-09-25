#!/usr/bin/env python3

import argparse
import sys
from dvrip import DVRIPCam
from pprint import pprint

# Telnet root password: xmhdipc

# A constant list of all settings to query when using the --dump option.
SETTINGS_TO_DUMP = [
    "fVideo.Tour", "fVideo.GUISet", "fVideo.WheelFunction", "fVideo.TVAdjust",
    "fVideo.AudioInFormat", "fVideo.Play", "fVideo.VideoOut", "fVideo.OSDWidget",
    "fVideo.Spot", "fVideo.Volume", "fVideo.LossShowStr", "fVideo.VideoOutPriority",
    "fVideo.VideoSeque", "fVideo.VoColorAdjust", "fVideo.OSDInfo", "fVideo.OsdLogo",
    "fVideo.VideoSignal", "fVideo.OEMChSeq", "fVideo.AudioSupportType", "Camera.Param",
    "Camera.ParamEx", "Camera.FishEye", "Camera.ClearFog", "Camera.MotorCtrl",
    "Camera.FishLensParam", "Camera.DistortionCorrect", "Camera.FishViCut",
    "Camera.WhiteLight", "AVEnc.CombineEncodeParam", "AVEnc.EncodeStaticParam",
    "AVEnc.Encode", "AVEnc.VideoWidget", "AVEnc.VideoColor", "AVEnc.CombineEncode",
    "AVEnc.CombineEncodeParam", "AVEnc.WaterMark", "AVEnc.EncodeStaticParamV2",
    "AVEnc.VideoColorCustom", "AVEnc.EncodeEx", "AVEnc.EncodeAddBeep"
]


def parse_ip_range(ip_string: str) -> list[str]:
    """
    Parses an IP range string (e.g., '192.168.1.10-15') into a list of IPs.
    Also handles single IP addresses.
    """
    if '-' not in ip_string:
        return [ip_string]

    try:
        parts = ip_string.split('-')
        base_ip_parts = parts[0].split('.')
        
        if len(base_ip_parts) != 4:
            raise ValueError("Invalid base IP format.")

        start_octet = int(base_ip_parts[3])
        end_octet = int(parts[1])
        prefix = ".".join(base_ip_parts[0:3])

        if not (0 <= start_octet <= 255 and 0 <= end_octet <= 255 and start_octet <= end_octet):
             raise ValueError("Invalid octet range.")

        return [f"{prefix}.{i}" for i in range(start_octet, end_octet + 1)]
    except (ValueError, IndexError) as e:
        sys.exit(f"Error: Invalid IP range format '{ip_string}'. Example: '192.168.76.71-77'. Details: {e}")


def set_compression(data: dict | list, value: str):
    """
    Recursively finds and sets the video compression value in a settings dictionary.
    """
    if isinstance(data, dict):
        if "Video" in data and "Compression" in data["Video"]:
            data["Video"]["Compression"] = value
        for v in data.values():
            set_compression(v, value)
    elif isinstance(data, list):
        for item in data:
            set_compression(item, value)


def dump_camera_settings(cam: DVRIPCam, ip: str):
    """
    Connects to a camera and pretty-prints all its settings.
    """
    print(f"--- Dumping all settings for {ip} ---")
    for setting in SETTINGS_TO_DUMP:
        try:
            print(f"\n#--- {setting} ---#")
            pprint(cam.get_info(setting))
        except Exception as e:
            print(f"Could not retrieve setting '{setting}': {e}")
    print(f"--- Finished dumping settings for {ip} ---")


def configure_camera(cam: DVRIPCam, ip: str, codec: str, gop: int, bitrate: int, no_reboot: bool):
    """
    Applies a standard configuration to a camera.
    """
    print(f"Configuring camera at {ip}...")

    # --- Set Camera Parameters ---
    cam_settings = cam.get_info("Camera.Param")
    cam_settings[0].update({
        'EsShutter': '0x00000000',
        'DayNightColor': '0x00000001',
        "GainParam": {"AutoGain": 1, "Gain": 50},
        "BroadTrends": {"AutoGain": 0, "Gain": 50},
        "ExposureParam": {"MostTime": '0x00010000'},
        "ExposureTime": '0x00000100',
        "LowLuxMode": 0,
        "LightRestrainLevel": 16,
        'WhiteBalance': '0x00000000',
        'AutomaticAdjustment': 3,
        'Day_nfLevel': 5,
        'Night_nfLevel': 5,
        'Ldc': 1
    })
    cam.set_info("Camera.Param", cam_settings)
    cam.set_info("Camera.ParamEx", cam_settings)
    print("-> Camera parameters set.")

    # --- Set Encoding Parameters ---
    enc_settings = cam.get_info("AVEnc.Encode")
    # Main Stream
    enc_settings[0]["MainFormat"][0]["Video"]["Resolution"] = '1080P'
    enc_settings[0]["MainFormat"][0]["Video"]["BitRate"] = bitrate
    enc_settings[0]["MainFormat"][0]["Video"]["Quality"] = 6
    enc_settings[0]["MainFormat"][0]["Video"]["GOP"] = gop
    # Extra Streams
    enc_settings[0]["ExtraFormat"][0]["Video"]["Resolution"] = 'HD1'
    enc_settings[0]["ExtraFormat"][1]["Video"]["Resolution"] = 'WSVGA'
    enc_settings[0]["ExtraFormat"][2]["Video"]["Resolution"] = 'WSVGA'
    # Apply compression setting to all streams
    set_compression(enc_settings, codec)
    cam.set_info("AVEnc.Encode", enc_settings)
    print(f"-> Encoding settings configured (Codec: {codec}, GOP: {gop}, Bitrate: {bitrate}).")

    # --- Set OSD/Widget Parameters ---
    widget_settings = cam.get_info("AVEnc.VideoWidget")
    widget_settings[0]["ChannelTitle"]["Name"] = "NMN"
    widget_settings[0]["ChannelTitleAttribute"]["EncodeBlend"] = False
    widget_settings[0]["Covers"][0]["EncodeBlend"] = True
    widget_settings[0]["Covers"][0]["RelativePos"] = [0, 7930, 1320, 8130]
    widget_settings[0]["TimeTitleAttribute"].update({
        "EncodeBlend": True,
        "PreviewBlend": True,
        "BackColor": "0xFF808080",
        "FrontColor": "0xFFFFFFFF",
        "RelativePos": [0, 8192, 0, 0]
    })
    cam.set_info("AVEnc.VideoWidget", widget_settings)
    print("-> Video widget (OSD) settings applied.")

    # --- Reboot Camera to Apply Changes (if not disabled) ---
    if not no_reboot:
        print(f"-> Rebooting {ip} to apply changes.")
        cam.reboot()
    else:
        print("-> --noreboot specified, skipping reboot.")


def main():
    """
    Main function to parse arguments and orchestrate camera configuration.
    """
    parser = argparse.ArgumentParser(
        description="Connect to and configure XM/DVRIP-based IP cameras.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.76.71-77",
        help="Target IP address or range.\nExamples:\n'192.168.76.71'\n'192.168.76.71-77'"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="HEVC",
        choices=["H.264", "HEVC"],
        help="Video codec to use. Default: HEVC"
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=4,
        help="GOP (Group of Pictures) value. Must be >= 1. Default: 4"
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=3072,
        help="Video bitrate in kbps. Must be a positive integer. Default: 3072"
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump all current settings from the camera(s) instead of configuring them."
    )
    parser.add_argument(
        "--noreboot",
        action="store_true",
        help="Do not reboot the camera after applying the configuration."
    )

    args = parser.parse_args()

    # --- Input Validation ---
    if args.gop < 1:
        sys.exit("Error: --gop must be an integer of at least 1.")
    if args.bitrate <= 0:
        sys.exit("Error: --bitrate must be a positive integer.")

    # --- Main Loop ---
    ips = parse_ip_range(args.ip)
    print(f"Targeting cameras: {ips}")

    for ip in ips:
        cam = None  # Ensure cam is defined for the finally block
        try:
            print(f"\nConnecting to {ip}...")
            cam = DVRIPCam(ip, user='admin', password='')
            if not cam.login():
                print(f"Error: Could not log in to {ip}. Check credentials and network. Skipping.")
                continue

            cam.set_time()

            if args.dump:
                dump_camera_settings(cam, ip)
            else:
                configure_camera(cam, ip, args.codec, args.gop, args.bitrate, args.noreboot)

        except Exception as e:
            print(f"An unexpected error occurred with camera {ip}: {e}")
        finally:
            if cam:
                cam.close()
                print(f"Connection to {ip} closed.")


if __name__ == "__main__":
    main()
