#!/usr/bin/env python3
"""
Camera Configuration Tool

A multi-purpose Python script for configuring DVRIP-based IP cameras.
It supports three modes of operation:
1. A command-line interface (CLI) for scripting and automation (--dump, --codec, etc.).
2. A graphical user interface (GUI) built with Tkinter (--gui).
3. A web-based interface built with Flask (--web).

The script is architected to share core camera communication logic between all modes.
"""

import argparse
import sys
from dvrip import DVRIPCam
from pprint import pprint
import threading
import json

# Telnet root password: xmhdipc

# Top-level setting groups used to discover all available subgroups on a camera.
MAIN_GROUPS = ["AVEnc", "Camera", "fVideo"]

# A dictionary containing documentation for specific camera parameters.
# Used by the GUI and Flask UIs to generate helpful descriptions and validate input.
DOCS = {
    # Camera.ParamEx
    'Camera.ParamEx.AeMeansure': {'desc': 'Sensitivity configuration.'},
    'Camera.ParamEx.AutomaticAdjustment': {'desc': 'Gear control.'},
    'Camera.ParamEx.BroadTrends.AutoGain': {
        'desc': 'Enable wide dynamic range (WDR).',
        'type': 'options', 'map': {'Off': '0', 'On': '1'}
    },
    'Camera.ParamEx.BroadTrends.Gain': {
        'desc': 'WDR upper limit.',
        'type': 'range', 'values': (0, 100)
    },
    'Camera.ParamEx.CorridorMode': {
        'desc': 'Switch corridor mode to rotate the screen.',
        'type': 'options', 'map': {
            'Normal mode': '0', 'Rotate 90° CCW': '1',
            'Rotate 180° CCW': '2', 'Rotate 270° CCW': '3'
        }
    },
    'Camera.ParamEx.Dis': {
        'desc': 'Electronic image stabilization.',
        'type': 'options', 'map': {'Off': '0', 'On': '1'}
    },
    'Camera.ParamEx.ExposureTime': {'desc': 'Effective exposure time.'},
    'Camera.ParamEx.Ldc': {
        'desc': 'Lens distortion correction.',
        'type': 'options', 'map': {'Off': '0', 'On': '1'}
    },
    'Camera.ParamEx.LowLuxMode': {
        'desc': 'Low-light mode.',
        'type': 'options', 'map': {'Off': '0', 'On': '1'}
    },
    'Camera.ParamEx.PreventOverExpo': {'desc': 'Prevent overexposure (face without red burst).'},
    'Camera.ParamEx.SoftPhotosensitivecontrol': {'desc': 'Soft light-sensitive control.'},
    'Camera.ParamEx.Style': {
        'desc': 'Image style.',
        'type': 'options', 'map': {'Style 1': 'typedefault', 'Style 2': 'type1', 'Style 3': 'type2'}
    },

    # Camera.Param
    'Camera.Param.AeSensitivity': {'desc': 'Sensitivity configuration.'},
    'Camera.Param.ApertureMode': {'desc': 'Configure automatic aperture.'},
    'Camera.Param.BLCMode': {'desc': 'Backlight compensation.'},
    'Camera.Param.DayNightColor': {'desc': 'Day-night mode.'},
    'Camera.Param.Day_nfLevel': {'desc': 'Daytime noise reduction level.'},
    'Camera.Param.Night_nfLevel': {'desc': 'Nighttime noise reduction level.'},
    'Camera.Param.DncThr': {'desc': 'Day-night conversion threshold.'},
    'Camera.Param.ElecLevel': {'desc': 'Set reference level value (auto exposure).'},
    'Camera.Param.EsShutter': {
        'desc': 'Electronic slow shutter.',
        'type': 'options', 'map': {'None': '0', 'Poor': '2', 'Medium': '4', 'Strong': '8'}
    },
    'Camera.Param.ExposureParam.LeastTime': {'desc': 'Minimum automatic exposure time.'},
    'Camera.Param.ExposureParam.Level': {'desc': 'Auto exposure mode or manual exposure level.'},
    'Camera.Param.ExposureParam.MostTime': {'desc': 'Maximum automatic exposure time.'},
    'Camera.Param.GainParam.AutoGain': {
        'desc': 'Auto gain enable.',
        'type': 'options', 'map': {'Off': '0', 'On': '1'}
    },
    'Camera.Param.GainParam.Gain': {
        'desc': 'Automatic gain upper limit (ceiling).',
        'type': 'range', 'values': (0, 100)
    },
    'Camera.Param.IRCUTMode': {'desc': 'IR-CUT switch mode.'},
    'Camera.Param.InfraredSwap': {
        'desc': 'Day night mode level polarity.',
        'type': 'options', 'map': {'Positive': '0', 'Negative': '1'}
    },
    'Camera.Param.IrcutSwap': {'desc': 'IR-CUT Sequence conversion.'},
    'Camera.Param.PictureFlip': {
        'desc': 'Flip the picture vertically.',
        'type': 'options', 'map': {'No Flip': '0x00000000', 'Flip': '0x00000001'}
    },
    'Camera.Param.PictureMirror': {
        'desc': 'Flip the picture horizontally (mirror mode).',
        'type': 'options', 'map': {'No Flip': '0x00000000', 'Flip': '0x00000001'}
    },
    'Camera.Param.RejectFlicker': {'desc': 'Anti-flicker function for fluorescent lamps.'},
    'Camera.Param.WhiteBalance': {'desc': 'White balance (scene mode).'}
}


class CameraController:
    """
    Handles all logic for camera communication, shared between UIs.
    This class caches camera data to avoid repeated network requests.
    """
    def __init__(self, ips):
        self.ips = ips
        self.all_camera_data = {}
        self.lock = threading.Lock()

    def load_camera_data(self, ip):
        if ip in self.all_camera_data:
            return self.all_camera_data[ip]

        with self.lock:
            if ip in self.all_camera_data:
                return self.all_camera_data[ip]

            cam_data = {}
            try:
                cam = DVRIPCam(ip, user='admin', password='')
                if not cam.login(): raise ConnectionError("Login failed.")
                for group in MAIN_GROUPS:
                    try:
                        subgroup_data = cam.get_info(group)
                        if isinstance(subgroup_data, dict):
                            prefixed_data = {f"{group}.{key}": value for key, value in subgroup_data.items()}
                            cam_data.update(prefixed_data)
                    except Exception: continue
                cam.close()
                self.all_camera_data[ip] = cam_data
                return cam_data
            except Exception as e:
                error_data = {"error": str(e)}
                self.all_camera_data[ip] = error_data
                return error_data

    def invalidate_cache(self, ip):
        """Removes an IP's data from the cache."""
        with self.lock:
            if ip in self.all_camera_data:
                del self.all_camera_data[ip]

    def apply_settings(self, ip, subgroup, new_settings):
        try:
            cam = DVRIPCam(ip, user='admin', password='')
            if not cam.login(): raise ConnectionError("Login failed.")
            cam.set_info(subgroup, new_settings)
            cam.close()
            return {"success": True, "message": f"Settings for '{subgroup}' applied to {ip}."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def reboot_camera(self, ip):
        try:
            cam = DVRIPCam(ip, user='admin', password='')
            if not cam.login(): raise ConnectionError("Login failed.")
            cam.reboot()
            cam.close()
            return {"success": True, "message": f"Reboot command sent to {ip}."}
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def set_camera_time(self, ip):
        try:
            cam = DVRIPCam(ip, user='admin', password='')
            if not cam.login(): raise ConnectionError("Login failed.")
            cam.set_time()
            cam.close()
            return {"success": True, "message": f"Time set successfully on {ip}."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def clone_settings(self, source_ip, dest_ip):
        """Clones all settings from source_ip to dest_ip."""
        source_data = self.load_camera_data(source_ip)
        
        if "error" in source_data:
            return {"success": False, "message": f"Could not load settings from source {source_ip}: {source_data['error']}"}

        errors = []
        groups_to_clone = sorted([group for group in source_data if group != "error"])
        
        for group_name in groups_to_clone:
            settings_data = source_data[group_name]
            if settings_data is None: continue # Skip empty groups
            
            result = self.apply_settings(dest_ip, group_name, settings_data)
            
            if not result["success"]:
                errors.append(f"Failed to apply '{group_name}': {result['message']}")

        self.invalidate_cache(dest_ip)
        
        if errors:
            return {"success": True, "message": "Clone complete (with errors)", "errors": errors}
        else:
            return {"success": True, "message": "Successfully cloned all settings."}


def main():
    """
    Main execution function. Parses arguments and launches the requested mode.
    """
    def parse_ip_range(ip_string: str) -> list[str]:
        if '-' not in ip_string: return [ip_string]
        try:
            parts = ip_string.split('-')
            base_ip_parts = parts[0].split('.')
            if len(base_ip_parts) != 4: raise ValueError("Invalid base IP format.")
            start_octet = int(base_ip_parts[3])
            end_octet = int(parts[1])
            prefix = ".".join(base_ip_parts[0:3])
            if not (0 <= start_octet <= 255 and 0 <= end_octet <= 255 and start_octet <= end_octet): raise ValueError("Invalid octet range.")
            return [f"{prefix}.{i}" for i in range(start_octet, end_octet + 1)]
        except (ValueError, IndexError) as e:
            sys.exit(f"Error: Invalid IP range format '{ip_string}'. Example: '192.168.76.71-77'. Details: {e}")

    def dump_camera_settings(cam: DVRIPCam, ip: str):
        print(f"--- Dumping all settings for {ip} ---")
        for group in MAIN_GROUPS:
            try:
                print(f"\n#--- Main Group: {group} ---#"), pprint(cam.get_info(group))
            except Exception as e:
                print(f"Could not retrieve main group '{group}': {e}")
        print(f"--- Finished dumping settings for {ip} ---")

    def set_compression(data: dict | list, value: str):
        if isinstance(data, dict):
            if "Video" in data and "Compression" in data["Video"]: data["Video"]["Compression"] = value
            for v in data.values(): set_compression(v, value)
        elif isinstance(data, list):
            for item in data: set_compression(item, value)

    def configure_camera(cam: DVRIPCam, ip: str, codec: str, gop: int, bitrate: int, no_reboot: bool):
        print(f"Configuring camera at {ip}...")
        try:
            cam_settings = cam.get_info("Camera.Param")
            cam_settings[0].update({'EsShutter': '0x00000000', 'DayNightColor': '0x00000001', "GainParam": {"AutoGain": 1, "Gain": 50}, "BroadTrends": {"AutoGain": 0, "Gain": 50}, "ExposureParam": {"MostTime": '0x00010000'}, "ExposureTime": '0x00000100', "LowLuxMode": 0, "LightRestrainLevel": 16, 'WhiteBalance': '0x00000000', 'AutomaticAdjustment': 3, 'Day_nfLevel': 5, 'Night_nfLevel': 5, 'Ldc': 1})
            cam.set_info("Camera.Param", cam_settings), cam.set_info("Camera.ParamEx", cam_settings)
            print("-> Camera parameters set.")
        except Exception as e:
            print(f"Warning: Could not set Camera.Param: {e}")
        try:
            enc_settings = cam.get_info("AVEnc.Encode")
            enc_settings[0]["MainFormat"][0]["Video"].update({"Resolution": '1080P', "BitRate": bitrate, "Quality": 6, "GOP": gop})
            enc_settings[0]["ExtraFormat"][0]["Video"]["Resolution"], enc_settings[0]["ExtraFormat"][1]["Video"]["Resolution"], enc_settings[0]["ExtraFormat"][2]["Video"]["Resolution"] = 'HD1', 'WSVGA', 'WSVGA'
            set_compression(enc_settings, codec)
            cam.set_info("AVEnc.Encode", enc_settings)
            print(f"-> Encoding settings configured (Codec: {codec}, GOP: {gop}, Bitrate: {bitrate}).")
        except Exception as e:
            print(f"Warning: Could not set AVEnc.Encode: {e}")
        if not no_reboot: print(f"Rebooting {ip}..."), cam.reboot()
        else: print("Skipping reboot as per --noreboot flag.")
    
    parser = argparse.ArgumentParser(description="Connect to and configure XM-based IP cameras.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ip", type=str, default="192.168.76.71-77", help="Target IP address or range.\nExamples:\n'192.168.76.71'\n'192.168.76.71-77'")
    parser.add_argument("--codec", type=str, default="H.265", choices=["H.264", "H.265"], help="CLI MODE: Video codec to use. Default: H.265")
    parser.add_argument("--gop", type=int, default=4, help="CLI MODE: GOP value. Must be >= 1. Default: 4")
    parser.add_argument("--bitrate", type=int, default=3072, help="CLI MODE: Video bitrate in kbps. Must be a positive integer. Default: 3072")
    parser.add_argument("--noreboot", action="store_true", help="CLI MODE: Do not reboot the camera after applying the configuration.")
    parser.add_argument("--port", type=int, default=5001, help="WEB MODE: Port for the web server. Default: 5001")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="Launch the Tkinter graphical user interface.")
    mode_group.add_argument("--dump", action="store_true", help="Dump all settings from the camera(s) to the console.")
    mode_group.add_argument("--web", action="store_true", help="Launch the web server interface.")
    
    args = parser.parse_args()
    
    try: 
        ips = parse_ip_range(args.ip)
    except SystemExit as e: 
        print(e)
        return

    controller = CameraController(ips)

    if args.gui:
        try:
            import tkinter as tk
            from tkinter import ttk, messagebox, font
        except ImportError:
            sys.exit("Tkinter is not available. Please install it to use --gui mode.")
        
        class CamConfigGUI:
            def __init__(self, controller):
                self.controller = controller
                self.ips = self.controller.ips
                self.ui_variables = {}
                self.root = tk.Tk()
                self.root.title("Camera Configuration Tool")
                self.root.geometry("1200x700")
                style = ttk.Style(self.root)
                style.configure("Even.TFrame", background="white")
                style.configure("Odd.TFrame", background="#f0f0f0")
                style.configure("Even.TLabel", background="white")
                style.configure("Odd.TLabel", background="#f0f0f0")
                style.configure("Even.TCheckbutton", background="white")
                style.configure("Odd.TCheckbutton", background="#f0f0f0")
                default_font = font.nametofont("TkDefaultFont")
                default_font.configure(size=9)
                self.italic_font = font.Font(family=default_font.cget("family"), size=default_font.cget("size"), slant="italic")
                self.root.option_add("*Font", default_font)
                
                # --- Top Control Panel (Fixed) ---
                top_frame = ttk.Frame(self.root, padding="5", relief=tk.RAISED, borderwidth=1)
                top_frame.pack(side="top", fill="x", pady=0, padx=0)
                top_frame.grid_columnconfigure(5, weight=1)

                ttk.Label(top_frame, text="Camera IP:").grid(row=0, column=0, padx=(0, 5))
                self.ip_var = tk.StringVar()
                self.ip_dropdown = ttk.Combobox(top_frame, textvariable=self.ip_var, state="readonly", width=15)
                self.ip_dropdown.grid(row=0, column=1, padx=5)
                self.ip_dropdown.bind("<<ComboboxSelected>>", self.on_camera_select)
                
                ttk.Label(top_frame, text="Main Group:").grid(row=0, column=2, padx=(10, 5))
                self.main_group_var = tk.StringVar()
                self.main_group_dropdown = ttk.Combobox(top_frame, textvariable=self.main_group_var, values=MAIN_GROUPS, state="readonly", width=10)
                self.main_group_dropdown.grid(row=0, column=3, padx=5)
                self.main_group_dropdown.bind("<<ComboboxSelected>>", self.on_main_group_select)
                
                ttk.Label(top_frame, text="Subgroup:").grid(row=0, column=4, padx=(10, 5))
                self.subgroup_var = tk.StringVar()
                self.subgroup_dropdown = ttk.Combobox(top_frame, textvariable=self.subgroup_var, state="readonly", width=25)
                self.subgroup_dropdown.grid(row=0, column=5, padx=5, sticky="ew")
                self.subgroup_dropdown.bind("<<ComboboxSelected>>", self.display_setting_group)
                
                ttk.Label(top_frame, text="Clone from:").grid(row=0, column=6, padx=(10, 5))
                self.clone_var = tk.StringVar()
                self.clone_dropdown = ttk.Combobox(top_frame, textvariable=self.clone_var, state="readonly", width=15)
                self.clone_dropdown.grid(row=0, column=7, padx=5)
                self.clone_dropdown.bind("<<ComboboxSelected>>", self.on_clone_select)
                
                self.reload_button = ttk.Button(top_frame, text="Reload", command=self.reload_camera_data)
                self.reload_button.grid(row=0, column=8, padx=(10, 5))
                self.set_time_button = ttk.Button(top_frame, text="Set Time", command=self.set_time)
                self.set_time_button.grid(row=0, column=9, padx=(5, 5))
                self.reboot_button = ttk.Button(top_frame, text="Reboot Camera", command=self.reboot_camera)
                self.reboot_button.grid(row=0, column=10, padx=(5, 5))
                self.quit_button = ttk.Button(top_frame, text="Quit", command=self.root.destroy)
                self.quit_button.grid(row=0, column=11, padx=(5, 5))
                
                # --- Scrollable Settings Area ---
                self.settings_outer_frame = ttk.Frame(self.root)
                self.settings_outer_frame.pack(expand=True, fill="both", padx=5, pady=(5,0))

                # --- Bottom Bar (Apply button and Status) ---
                bottom_frame = ttk.Frame(self.root)
                bottom_frame.pack(side="bottom", fill="x")
                self.apply_button = ttk.Button(bottom_frame, text="Apply Changes", state="disabled", command=self.apply_settings)
                self.apply_button.pack(side="top", fill="x", padx=5, pady=5)
                self.status_var = tk.StringVar(value="Ready.")
                status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w', padding=2)
                status_bar.pack(side="bottom", fill="x", padx=5, pady=(0,5))
                
                # --- Initial Population ---
                self.ip_dropdown['values'] = self.ips
                if self.ips:
                    self.ip_dropdown.set(self.ips[0])
                    self.on_camera_select()
                self.root.mainloop()

            def _update_status(self, text):
                self.status_var.set(text)
                self.root.update_idletasks()
            def _on_mousewheel(self, event):
                if hasattr(self, 'canvas'):
                    if event.num == 0 or event.delta:
                        scroll_delta = -1 * (event.delta // 120) if sys.platform == 'win32' else -1 * event.delta
                        self.canvas.yview_scroll(scroll_delta, "units")
                    elif event.num == 4: self.canvas.yview_scroll(-1, "units")
                    elif event.num == 5: self.canvas.yview_scroll(1, "units")
            def _bind_mousewheel_recursively(self, widget):
                widget.bind("<MouseWheel>", self._on_mousewheel); widget.bind("<Button-4>", self._on_mousewheel); widget.bind("<Button-5>", self._on_mousewheel)
                for child in widget.winfo_children(): self._bind_mousewheel_recursively(child)
            def _adjust_value(self, var, amount):
                original_str = var.get(); is_hex = original_str.lower().startswith('0x')
                try:
                    if is_hex:
                        width = len(original_str) - 2; val = int(original_str, 16); new_val = max(0, val + amount); var.set(f'0x{new_val:0{width}x}')
                    else:
                        val = int(original_str); new_val = val + amount; var.set(str(new_val))
                except ValueError: return
            def reload_camera_data(self):
                selected_ip = self.ip_var.get()
                if not selected_ip:
                    messagebox.showerror("Error", "No camera selected.")
                    return

                # Store current selections
                current_subgroup = self.subgroup_var.get()

                self._update_status(f"Reloading configuration for {selected_ip}...")
                self.root.config(cursor="watch")
                
                self.controller.invalidate_cache(selected_ip)
                data = self.controller.load_camera_data(selected_ip)
                
                self.root.config(cursor="")

                if "error" in data:
                    messagebox.showerror("Connection Error", f"Could not reload settings for {selected_ip}:\n{data['error']}")
                    self._update_status(f"Error: Failed to reload settings for {selected_ip}.")
                    # On error, fall back to the default behavior
                    self.on_camera_select()
                else:
                    self._update_status(f"Successfully reloaded settings for {selected_ip}.")
                    # The main group is preserved, so just update the subgroups and try to restore selection
                    self.update_subgroup_dropdown(preferred_subgroup=current_subgroup)

            def on_camera_select(self, event=None):
                selected_ip = self.ip_var.get()
                if not selected_ip: return
                self.update_clone_dropdown()
                self._update_status(f"Loading settings for {selected_ip}...")
                self.root.config(cursor="watch")
                data = self.controller.load_camera_data(selected_ip)
                self.root.config(cursor="")
                if "error" in data:
                    messagebox.showerror("Connection Error", f"Could not load settings for {selected_ip}:\n{data['error']}")
                    self._update_status(f"Error: Failed to load settings for {selected_ip}.")
                else: self._update_status(f"Successfully loaded settings for {selected_ip}.")
                self.main_group_dropdown.set(MAIN_GROUPS[0]); self.update_subgroup_dropdown()
            def on_main_group_select(self, event=None): self.update_subgroup_dropdown()
            def update_clone_dropdown(self):
                """Updates the 'Clone from:' dropdown to show all IPs except the selected one."""
                selected_ip = self.ip_var.get()
                other_ips = [ip for ip in self.ips if ip != selected_ip]
                self.clone_dropdown['values'] = other_ips
                self.clone_var.set("")
            def update_subgroup_dropdown(self, preferred_subgroup=None):
                selected_ip = self.ip_var.get(); main_group = self.main_group_var.get()
                camera_settings = self.controller.all_camera_data.get(selected_ip, {})
                if "error" in camera_settings:
                    self.subgroup_dropdown['values'], self.subgroup_dropdown.set([], ''); self.clear_settings_frame(); return
                subgroups = sorted([k for k, v in camera_settings.items() if k.startswith(main_group + '.') and v])
                self.subgroup_dropdown['values'] = subgroups
                
                if preferred_subgroup and preferred_subgroup in subgroups:
                    self.subgroup_dropdown.set(preferred_subgroup)
                else:
                    self.subgroup_dropdown.set(subgroups[0] if subgroups else '')
                
                self.display_setting_group()
            def clear_settings_frame(self):
                for widget in self.settings_outer_frame.winfo_children(): widget.destroy()
            def display_setting_group(self, event=None):
                self.clear_settings_frame(); self.apply_button.config(state="disabled")
                subgroup = self.subgroup_var.get()
                if not subgroup: return
                selected_ip = self.ip_var.get(); data = self.controller.all_camera_data.get(selected_ip, {}).get(subgroup)
                if data is None: return
                
                self.canvas = tk.Canvas(self.settings_outer_frame, borderwidth=0, highlightthickness=0)
                # --- Pack and then immediately HIDE the canvas ---
                self.canvas.pack(side="left", fill="both", expand=True)
                self.canvas.pack_forget() # <-- HIDE CANVAS
                
                scrollbar = ttk.Scrollbar(self.settings_outer_frame, orient="vertical", command=self.canvas.yview)
                scrollbar.pack(side="right", fill="y"); self.canvas.configure(yscrollcommand=scrollbar.set)
                scrollable_frame = ttk.Frame(self.canvas)
                self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

                # Store the handler, but don't bind it just yet
                configure_handler = lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                
                # Unbind temporarily to prevent recalculation on every widget add
                scrollable_frame.unbind("<Configure>")

                self.ui_variables = {}; display_data = data[0] if isinstance(data, list) and data else data
                self._populate_top_level(scrollable_frame, display_data, self.ui_variables, path_prefix=subgroup)

                # Force Tkinter to process all the new widgets (this is fast as it's not drawing)
                self.root.update_idletasks() 
                # Now, calculate the scrollregion just *once*
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                # Re-bind the handler for future window resizes
                scrollable_frame.bind("<Configure>", configure_handler)
                
                self._bind_mousewheel_recursively(scrollable_frame); self.canvas.bind("<MouseWheel>", self._on_mousewheel); self.canvas.bind("<Button-4>", self._on_mousewheel); self.canvas.bind("<Button-5>", self._on_mousewheel)

                # --- Now, SHOW the fully populated canvas ---
                self.canvas.pack(side="left", fill="both", expand=True) # <-- SHOW CANVAS
                
                self._bind_mousewheel_recursively(scrollable_frame); self.canvas.bind("<MouseWheel>", self._on_mousewheel); self.canvas.bind("<Button-4>", self._on_mousewheel); self.canvas.bind("<Button-5>", self._on_mousewheel)
            def _populate_top_level(self, parent, data, ui_vars_dict, path_prefix=""):
                parent.columnconfigure(0, weight=1)
                if isinstance(data, dict):
                    for i, (key, value) in enumerate(data.items()):
                        style_prefix = "Odd" if i % 2 else "Even"
                        group_frame = ttk.Frame(parent, style=f"{style_prefix}.TFrame", padding=5)
                        group_frame.grid(row=i, column=0, sticky="ew")
                        ui_vars_dict[key] = {}
                        self._populate_frame_with_widgets(group_frame, {key: value}, ui_vars_dict, path_prefix=path_prefix, style_prefix=style_prefix)
            def _populate_frame_with_widgets(self, parent, data, ui_vars_dict, row=0, indent=0, path_prefix="", style_prefix=""):
                parent.grid_columnconfigure(indent, weight=0); parent.grid_columnconfigure(indent + 1, weight=1); parent.grid_columnconfigure(indent + 2, weight=2)
                if isinstance(data, dict):
                    for key, value in data.items():
                        full_path = f"{path_prefix}.{key}" if path_prefix else key
                        label = ttk.Label(parent, text=str(key), style=f"{style_prefix}.TLabel")
                        label.grid(row=row, column=indent, sticky='nw', padx=(indent * 20, 5), pady=2)
                        doc = DOCS.get(full_path)
                        should_recurse = False
                        if isinstance(value, dict):
                            should_recurse = True
                        elif isinstance(value, list) and value and isinstance(value[0], dict):
                            # Only recurse for lists of dictionaries, not simple lists (like RelativePos)
                            should_recurse = True

                        if should_recurse:
                            if doc and 'desc' in doc:
                                desc_label = ttk.Label(parent, text=doc['desc'], style=f"{style_prefix}.TLabel", font=self.italic_font, wraplength=400, justify='left')
                                desc_label.grid(row=row, column=indent + 2, sticky='nw', pady=2, padx=5)
                            ui_vars_dict[key] = {}; row = self._populate_frame_with_widgets(parent, value, ui_vars_dict[key], row + 1, indent + 1, path_prefix=full_path, style_prefix=style_prefix)
                        else:
                            widget_info, var = {}, None; is_numeric = isinstance(value, int) or (isinstance(value, str) and value.lower().startswith('0x'))
                            if doc and doc.get('type') == 'options':
                                var = tk.StringVar(); rev_map = {v: k for k, v in doc['map'].items()}
                                widget = ttk.Combobox(parent, textvariable=var, state='readonly', width=20)
                                widget['values'] = list(doc['map'].keys()); var.set(rev_map.get(str(value), str(value)))
                                widget_info.update({'type': 'combobox', 'map': doc['map'], 'var': var}); widget.grid(row=row, column=indent + 1, sticky='w', pady=2)
                            elif isinstance(value, bool):
                                var = tk.BooleanVar(value=value); widget = ttk.Checkbutton(parent, variable=var, style=f"{style_prefix}.TCheckbutton")
                                widget_info.update({'type': 'checkbox', 'var': var}); widget.grid(row=row, column=indent + 1, sticky='w', pady=2)
                            elif is_numeric:
                                spinbox_frame = ttk.Frame(parent, style=f"{style_prefix}.TFrame")
                                spinbox_frame.grid(row=row, column=indent + 1, sticky='ew', pady=0); spinbox_frame.columnconfigure(0, weight=1)
                                var = tk.StringVar(value=str(value)); widget = ttk.Entry(spinbox_frame, textvariable=var)
                                up_button = ttk.Button(spinbox_frame, text="+", width=2, command=lambda v=var: self._adjust_value(v, 1))
                                down_button = ttk.Button(spinbox_frame, text="-", width=2, command=lambda v=var: self._adjust_value(v, -1))
                                widget.grid(row=0, column=0, sticky='ew'), up_button.grid(row=0, column=1), down_button.grid(row=0, column=2)
                                widget_info.update({'type': 'entry', 'var': var})
                            else:
                                var = tk.StringVar(value=str(value)); widget = ttk.Entry(parent, textvariable=var)
                                widget_info.update({'type': 'entry', 'var': var}); widget.grid(row=row, column=indent + 1, sticky='ew', pady=2)
                            if doc and 'desc' in doc:
                                desc_label = ttk.Label(parent, text=doc['desc'], style=f"{style_prefix}.TLabel", font=self.italic_font, wraplength=400, justify='left')
                                desc_label.grid(row=row, column=indent + 2, sticky='nw', pady=2, padx=5)
                            if isinstance(value, bool): widget.config(command=lambda *args: self.apply_button.config(state="normal"))
                            else: var.trace_add("write", lambda *args: self.apply_button.config(state="normal"))
                            ui_vars_dict[key] = widget_info
                            row += 1
                elif isinstance(data, list):
                    ui_vars_dict['_list_data_'] = []
                    for i, item in enumerate(data):
                        idx_label = ttk.Label(parent, text=f"[{i}]", style=f"{style_prefix}.TLabel")
                        idx_label.grid(row=row, column=indent, sticky='nw', padx=(indent * 20, 5), pady=2)
                        item_vars = {}; ui_vars_dict['_list_data_'].append(item_vars)
                        row = self._populate_frame_with_widgets(parent, item, item_vars, row + 1, indent + 1, path_prefix=f"{path_prefix}.[{i}]", style_prefix=style_prefix)
                return row
            def apply_settings(self):
                subgroup = self.subgroup_var.get()
                if not subgroup: return
                selected_ip = self.ip_var.get()
                self._update_status(f"Applying '{subgroup}' to {selected_ip}...")
                original_data = self.controller.all_camera_data[selected_ip][subgroup]
                new_settings = self._reconstruct_settings(self.ui_variables, original_data)
                result = self.controller.apply_settings(selected_ip, subgroup, new_settings)
                if result["success"]:
                    self._update_status(f"Settings applied. Reloading from {selected_ip} to verify...")
                    self.controller.invalidate_cache(selected_ip)
                    reloaded_data = self.controller.load_camera_data(selected_ip)
                    if "error" in reloaded_data:
                        messagebox.showwarning("Verification Failed", f"Settings applied successfully to {selected_ip}, but verification by reloading failed:\n{reloaded_data['error']}")
                        self._update_status(f"Error: Could not verify settings for {selected_ip}.")
                    else:
                        messagebox.showinfo("Success", result["message"])
                        self.display_setting_group()
                        self._update_status(f"Successfully applied and verified settings for {selected_ip}.")
                else:
                    messagebox.showerror("Error", f"Failed to apply settings:\n{result['message']}")
                    self._update_status("Error: Failed to apply settings.")
            def reboot_camera(self):
                selected_ip = self.ip_var.get()
                if not selected_ip: messagebox.showerror("Error", "No camera selected.")
                if messagebox.askyesno("Confirm Reboot", f"Are you sure you want to reboot the camera at {selected_ip}?"):
                    self._update_status(f"Sending reboot command to {selected_ip}...")
                    result = self.controller.reboot_camera(selected_ip)
                    if result["success"]: messagebox.showinfo("Success", "Reboot command sent."); self._update_status("Reboot command sent successfully.")
                    else: messagebox.showerror("Error", f"Failed to send reboot command: {result['message']}"); self._update_status("Error: Failed to send reboot command.")
            def set_time(self):
                selected_ip = self.ip_var.get()
                if not selected_ip: messagebox.showerror("Error", "No camera selected.")
                if messagebox.askyesno("Confirm Set Time", f"Are you sure you want to sync the time on {selected_ip}?"):
                    self._update_status(f"Setting time on {selected_ip}...")
                    result = self.controller.set_camera_time(selected_ip)
                    if result["success"]: messagebox.showinfo("Success", result["message"]); self._update_status(result["message"])
                    else: messagebox.showerror("Error", f"Failed to set time:\n{result['message']}"); self._update_status("Error: Failed to set time.")

            def on_clone_select(self, event=None):
                source_ip = self.clone_var.get()
                dest_ip = self.ip_var.get()
                
                if not source_ip or not dest_ip:
                    return

                if not messagebox.askyesno("Confirm Clone", f"Are you sure you want to clone all settings from\n{source_ip}\nto\n{dest_ip}?\n\nThis will overwrite all settings on {dest_ip}."):
                    self.clone_var.set("") # Reset dropdown
                    return

                self._update_status(f"Cloning settings from {source_ip} to {dest_ip}...")
                self.root.config(cursor="watch")

                # 1. Use the controller's clone method
                result = self.controller.clone_settings(source_ip, dest_ip)
                
                # 2. Report results
                self.root.config(cursor="")
                self.clone_var.set("") # Reset dropdown

                if "errors" in result and result["errors"]:
                    messagebox.showwarning("Clone Complete (with errors)", f"Finished cloning from {source_ip} to {dest_ip}, but some errors occurred:\n\n" + "\n".join(result["errors"]))
                elif not result["success"]:
                     messagebox.showerror("Clone Error", f"Could not clone settings from {source_ip}:\n{result['message']}")
                else:
                    messagebox.showinfo("Clone Complete", f"Successfully cloned all settings from {source_ip} to {dest_ip}.")

                # 3. Refresh the destination camera's GUI to show the new settings
                # The controller method already invalidated the cache
                self._update_status(f"Reloading {dest_ip} to reflect new settings...")
                self.reload_camera_data()

            def _reconstruct_settings(self, ui_vars, original_data):
                if isinstance(original_data, list):
                    if '_list_data_' in ui_vars:
                        # This is a list of items we have UI for (e.g., 'Covers')
                        new_list = []
                        list_vars = ui_vars['_list_data_']
                        # We must iterate over list_vars, which was built from original_data
                        # and is the source of truth for the UI.
                        for i, item_vars in enumerate(list_vars):
                            if i < len(original_data):
                                # Reconstruct this item using its UI vars and original data
                                new_list.append(self._reconstruct_settings(item_vars, original_data[i]))
                            else:
                                # Fallback: if original_data is somehow shorter,
                                # use the last good item as a template.
                                if original_data:
                                     new_list.append(self._reconstruct_settings(item_vars, original_data[-1]))
                        return new_list
                    elif original_data:
                        # This is a list wrapper, like 'VideoWidget' itself,
                        # for which we did *not* build a '_list_data_'
                        return [self._reconstruct_settings(ui_vars, original_data[0])]
                    else:
                        # This is an empty list
                        return []
                new_dict = {}
                for key, value in original_data.items():
                    ui_info = ui_vars.get(key)
                    if not ui_info: 
                        new_dict[key] = value
                        continue

                    # Check if ui_info is a widget-describing dict or a nested structure dict
                    if 'type' in ui_info:
                        # It's a widget
                        if ui_info['type'] == 'combobox':
                            display_val = ui_info['var'].get(); new_dict[key] = ui_info['map'].get(display_val, display_val)
                        elif ui_info['type'] == 'checkbox': new_dict[key] = ui_info['var'].get()
                        elif ui_info['type'] == 'entry':
                            original_type = type(value); str_val = ui_info['var'].get()
                            try:
                                if isinstance(value, str) and value.lower().startswith('0x'): new_dict[key] = str_val
                                elif original_type is bool: new_dict[key] = str_val.lower() in ['true', '1', 't', 'y', 'yes']
                                
                                elif original_type is list:
                                    try:
                                        # Try to parse it as a JSON list (e.g., [10, 20, 30, 40])
                                        new_val = json.loads(str_val)
                                        new_dict[key] = new_val if isinstance(new_val, list) else str_val
                                    except json.JSONDecodeError:
                                        new_dict[key] = str_val # Save as string if not valid JSON
                                
                                else: new_dict[key] = original_type(str_val)
                            except (ValueError, TypeError): new_dict[key] = str_val
                    
                    elif isinstance(ui_info, dict): 
                        # It's a nested structure, recurse
                        new_dict[key] = self._reconstruct_settings(ui_info, value)
                    
                    else:
                        # Fallback, though this shouldn't be hit with current logic
                        new_dict[key] = value 
                        
                return new_dict
        
        print("Launching GUI...")
        CamConfigGUI(controller)

    elif args.web:
        try:
            from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
            import simplepam
            import os
        except ImportError:
            sys.exit("Flask or simplepam is not available. Please install them to use --web mode (pip install flask simplepam).")

        app = Flask(__name__)
        app.secret_key = os.urandom(24)

        LOGIN_TEMPLATE = """
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Login - Camera Config</title>
        <style>body{font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;background-color:#f0f0f0}form{background:white;padding:2rem;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,.1)}h1{text-align:center}.form-group{margin-bottom:1rem}label{display:block;margin-bottom:.5rem}input{width:100%;padding:.5rem;border:1px solid #ccc;border-radius:4px}button{width:100%;padding:.75rem;border:none;background-color:#007bff;color:white;border-radius:4px;cursor:pointer}.error{color:red;text-align:center;margin-top:1rem}</style></head>
        <body><form method="POST"><h1>System Login</h1>{% if error %}<p class="error">{{ error }}</p>{% endif %}<div class="form-group"><label for="username">Username</label><input type="text" id="username" name="username" required></div><div class="form-group"><label for="password">Password</label><input type="password" id="password" name="password" required></div><button type="submit">Login</button></form></body></html>
        """

        MAIN_TEMPLATE = r"""
        <!DOCTYPE html><html><head><title>Camera Config</title><meta charset="UTF-8">
        <style>body{font-family:sans-serif;margin:0;background:#f8f9fa;color:#333}.top-bar{background:#fff;padding:10px 20px;border-bottom:1px solid #ddd;display:flex;align-items:center;gap:15px;position:sticky;top:0;z-index:1000}.top-bar label{font-weight:bold}.top-bar select,.top-bar button{padding:8px;border-radius:4px;border:1px solid #ccc}.content{padding:20px}#settings-view{background:white;border:1px solid #ddd;border-radius:4px}.group-frame{padding:10px;border-bottom:1px solid #ddd}.group-frame:last-child{border-bottom:none}.group-frame.even{background-color:#fff}.group-frame.odd{background-color:#f0f0f0}.row{display:grid;grid-template-columns:250px 1fr 2fr;align-items:center;gap:10px;padding:5px 0}.row-indent-1{margin-left:20px}.row-indent-2{margin-left:40px}.description{font-style:italic;color:#555;font-size:.9em}.spinbox{display:flex}.spinbox input{flex-grow:1}.spinbox button{width:30px}#status-bar{position:fixed;bottom:0;left:0;width:100%;background:#333;color:white;padding:5px 10px;font-size:.9em}</style></head>
        <body><div class="top-bar"><label for="ip-select">Camera IP:</label><select id="ip-select">{% for ip in ips %}<option value="{{ ip }}">{{ ip }}</option>{% endfor %}</select><label for="main-group-select">Main Group:</label><select id="main-group-select">{% for group in main_groups %}<option value="{{ group }}">{{ group }}</option>{% endfor %}</select><label for="subgroup-select">Subgroup:</label><select id="subgroup-select"></select><label for="clone-select">Clone from:</label><select id="clone-select"><option value="">-- Select Source --</option></select><button id="reload-btn">Reload</button><button id="set-time-btn">Set Time</button><button id="reboot-btn">Reboot Camera</button><button id="apply-btn" disabled>Apply Changes</button></div><div class="content"><div id="settings-view"></div></div><div id="status-bar">Ready.</div>
        <script>
            const ipSelect = document.getElementById('ip-select'),
                  mainGroupSelect = document.getElementById('main-group-select'),
                  subgroupSelect = document.getElementById('subgroup-select'),
                  cloneSelect = document.getElementById('clone-select'),
                  settingsView = document.getElementById('settings-view'),
                  applyBtn = document.getElementById('apply-btn'),
                  rebootBtn = document.getElementById('reboot-btn'),
                  setTimeBtn = document.getElementById('set-time-btn'),
                  reloadBtn = document.getElementById('reload-btn'),
                  statusBar = document.getElementById('status-bar'),
                  DOCS = {{ docs | tojson | safe }};
            let cameraData = {},
                originalSubgroupData = {},
                isListWrapped = false,
                allIps = {{ ips | tojson | safe }};

            function updateStatus(text) { statusBar.textContent = text; }
            
            function updateCloneSelect() {
                const selectedIp = ipSelect.value;
                cloneSelect.innerHTML = '<option value="">-- Select Source --</option>'; // Reset
                
                const otherIps = allIps.filter(ip => ip !== selectedIp);
                otherIps.forEach(ip => {
                    const option = document.createElement('option');
                    option.value = ip;
                    option.textContent = ip;
                    cloneSelect.appendChild(option);
                });
                cloneSelect.value = ""; // Set to default
            }

            async function reloadCameraData() {
                const ip = ipSelect.value;
                if (!ip) return;

                const currentSubgroup = subgroupSelect.value;

                updateStatus(`Reloading configuration for ${ip}...`);
                delete cameraData[ip];

                try {
                    const response = await fetch(`/api/data/${ip}`);
                    const data = await response.json();
                    if (data.error) throw new Error(data.error);
                    cameraData[ip] = data;
                    updateStatus(`Successfully reloaded settings for ${ip}.`);
                    // On success, update UI preserving subgroup
                    updateSubgroupSelect(currentSubgroup);
                } catch (e) {
                    cameraData[ip] = { error: e.message };
                    updateStatus(`Error: Failed to reload settings for ${ip}.`);
                    // On error, just update the UI which will show the error message
                    updateSubgroupSelect();
                }
            }
            async function onIpSelectChange() {
                const ip = ipSelect.value;
                if (!ip) return;
                
                updateCloneSelect(); // Update clone dropdown
                
                updateStatus(`Loading settings for ${ip}...`);
                if (!cameraData[ip]) {
                    try {
                        const response = await fetch(`/api/data/${ip}`);
                        const data = await response.json();
                        if (data.error) throw new Error(data.error);
                        cameraData[ip] = data;
                        updateStatus(`Successfully loaded settings for ${ip}.`);
                    } catch (e) {
                        cameraData[ip] = { error: e.message };
                        updateStatus(`Error: Failed to load settings for ${ip}.`);
                    }
                }
                updateSubgroupSelect();
            }
            function updateSubgroupSelect(preferredSubgroup) {
                const ip = ipSelect.value, mainGroup = mainGroupSelect.value, settings = cameraData[ip] || {};
                subgroupSelect.innerHTML = '';
                if (settings.error) { 
                    settingsView.innerHTML = `<p style="color:red;">${settings.error}</p>`; 
                    onSubgroupChange(); // Clear the view
                    return; 
                }
                const subgroups = Object.keys(settings).filter(k => k.startsWith(mainGroup + '.') && settings[k]).sort();
                
                if (subgroups.length === 0) {
                    onSubgroupChange(); // Clear the view
                    return;
                }

                subgroups.forEach(sg => { 
                    const option = document.createElement('option'); 
                    option.value = sg; 
                    option.textContent = sg; 
                    subgroupSelect.appendChild(option); 
                });

                if (preferredSubgroup && subgroups.includes(preferredSubgroup)) {
                    subgroupSelect.value = preferredSubgroup;
                } else {
                    subgroupSelect.value = subgroups[0];
                }
                
                onSubgroupChange();
            }
            function onSubgroupChange() {
                const ip = ipSelect.value, subgroup = subgroupSelect.value;
                settingsView.innerHTML = '';
                applyBtn.disabled = true;
                if (!subgroup) return;
                originalSubgroupData = (cameraData[ip] || {})[subgroup];
                isListWrapped = Array.isArray(originalSubgroupData);
                const displayData = isListWrapped && originalSubgroupData.length > 0 ? originalSubgroupData[0] : originalSubgroupData;
                renderSettings(displayData, subgroup);
            }
            function renderSettings(data, subgroup) {
                if (!data || typeof data !== 'object') return;
                let i = 0;
                for (const key in data) {
                    const groupFrame = document.createElement('div');
                    groupFrame.className = `group-frame ${i++ % 2 ? 'odd' : 'even'}`;
                    _renderValue(groupFrame, key, data[key], `${subgroup}.${key}`, 0, key);
                    settingsView.appendChild(groupFrame);
                }
            }
            function _renderValue(parent, key, value, fullPath, indent, relPath) {
                const row = document.createElement('div');
                row.className = `row row-indent-${indent}`;
                const doc = DOCS[fullPath] || {};
                const label = document.createElement('label');
                label.textContent = key; row.appendChild(label);
                
                if (typeof value === 'object' && value !== null) {
                    row.appendChild(document.createElement('div'));
                    const desc = document.createElement('div');
                    desc.className = 'description';
                    if(doc.desc) desc.textContent = doc.desc;
                    row.appendChild(desc);
                    parent.appendChild(row);
                    if (Array.isArray(value)) {
                        value.forEach((item, i) => { _renderValue(parent, `[${i}]`, item, `${fullPath}[${i}]`, indent + 1, `${relPath}.${i}`); });
                    } else {
                        for (const subKey in value) { _renderValue(parent, subKey, value[subKey], `${fullPath}.${subKey}`, indent + 1, `${relPath}.${subKey}`); }
                    }
                } else {
                    const widgetContainer = document.createElement('div');
                    if (doc.type === "options") {
                        const select = document.createElement('select');
                        select.dataset.path = relPath;
                        for (const opt in doc.map) { const option = document.createElement('option'); option.value = doc.map[opt]; option.textContent = opt; select.appendChild(option); }
                        select.value = value;
                        widgetContainer.appendChild(select);
                    } else if (typeof value === 'boolean') {
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox'; checkbox.checked = value; checkbox.dataset.path = relPath;
                        widgetContainer.appendChild(checkbox);
                    } else if (typeof value === 'number' || (typeof value === 'string' && value.startsWith('0x'))) {
                        const isHex = typeof value === 'string' && value.startsWith('0x');
                        const spinbox = document.createElement('div'); spinbox.className = 'spinbox';
                        const input = document.createElement('input'); input.type = 'text'; input.value = value; input.dataset.path = relPath;
                        const upBtn = document.createElement('button'); upBtn.textContent = '+';
                        const downBtn = document.createElement('button'); downBtn.textContent = '-';
                        upBtn.onclick = () => adjustValue(input, 1, isHex);
                        downBtn.onclick = () => adjustValue(input, -1, isHex);
                        spinbox.append(input, upBtn, downBtn);
                        widgetContainer.appendChild(spinbox);
                    } else {
                        const input = document.createElement('input'); input.type = 'text'; input.value = value; input.dataset.path = relPath;
                        widgetContainer.appendChild(input);
                    }
                    row.appendChild(widgetContainer);
                    const desc = document.createElement('div');
                    desc.className = 'description';
                    if (doc.desc) desc.textContent = doc.desc;
                    row.appendChild(desc);
                    parent.appendChild(row);
                }
            }
            function adjustValue(input, amount, isHex) {
                const originalStr = input.value;
                try {
                    if (isHex) {
                        let width = originalStr.length - 2;
                        let val = parseInt(originalStr, 16);
                        let newVal = Math.max(0, val + amount);
                        input.value = `0x${newVal.toString(16).padStart(width, '0')}`;
                    } else {
                        input.value = parseInt(originalStr, 10) + amount;
                    }
                    applyBtn.disabled = false;
                } catch (e) {}
            }
            async function applySettings() {
                const ip = ipSelect.value, subgroup = subgroupSelect.value;
                const newSettings = JSON.parse(JSON.stringify(originalSubgroupData));
                const settingsToApply = isListWrapped ? newSettings[0] : newSettings;
                
                function updateValueByPath(obj, path, value) {
                    const keys = path.replace(/\[(\d+)\]/g, '.$1').split('.');
                    let temp = obj;
                    for (let i = 0; i < keys.length - 1; i++) {
                        if (temp[keys[i]] === undefined) return;
                        temp = temp[keys[i]];
                    }
                    const finalKey = keys[keys.length - 1];
                    const originalValue = temp[finalKey];
                    if (typeof originalValue === 'boolean') temp[finalKey] = value;
                    else if (typeof originalValue === 'number' && !originalValue.toString().startsWith('0x')) temp[finalKey] = Number(value);
                    else temp[finalKey] = value;
                }
                settingsView.querySelectorAll('[data-path]').forEach(el => {
                    let value = el.type === 'checkbox' ? el.checked : el.value;
                    updateValueByPath(settingsToApply, el.dataset.path, value);
                });
                
                updateStatus(`Applying '${subgroup}' to ${ip}...`);
                try {
                    const response = await fetch(`/api/apply/${ip}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ subgroup: subgroup, settings: newSettings }) });
                    const result = await response.json();
                    if (!result.success) throw new Error(result.message);
                    
                    if (result.reloaded_data) {
                        cameraData[ip] = result.reloaded_data;
                        updateStatus("Settings applied and verified successfully.");
                    } else {
                        delete cameraData[ip]; // Invalidate cache to force reload next time
                        updateStatus("Settings applied, but verification by reloading failed. Data may be stale.");
                    }
                    onSubgroupChange();
                    applyBtn.disabled = true;
                } catch (e) { updateStatus(`Error: ${e.message}`); }
            }
            async function rebootCamera() {
                const ip = ipSelect.value;
                if (!ip || !confirm(`Are you sure you want to reboot the camera at ${ip}?`)) return;
                updateStatus(`Sending reboot command to ${ip}...`);
                try {
                    const response = await fetch(`/api/reboot/${ip}`, { method: 'POST' });
                    const result = await response.json();
                    if (!result.success) throw new Error(result.message);
                    updateStatus(result.message);
                } catch (e) { updateStatus(`Error: ${e.message}`); }
            }
            async function setTime() {
                const ip = ipSelect.value;
                if (!ip || !confirm(`Are you sure you want to sync time on ${ip}?`)) return;
                updateStatus(`Setting time on ${ip}...`);
                try {
                    const response = await fetch(`/api/set_time/${ip}`, { method: 'POST' });
                    const result = await response.json();
                    if (!result.success) throw new Error(result.message);
                    alert(result.message);
                    updateStatus(result.message);
                } catch (e) { alert(`Error: ${e.message}`); updateStatus(`Error: ${e.message}`); }
            }
            
            async function onCloneSelect() {
                const sourceIp = cloneSelect.value;
                const destIp = ipSelect.value;

                if (!sourceIp) return; // Do nothing if the default "-- Select Source --" is chosen

                if (!confirm(`Are you sure you want to clone all settings from\n${sourceIp}\nto\n${destIp}?\n\nThis will overwrite all settings on ${destIp}.`)) {
                    cloneSelect.value = ""; // Reset dropdown
                    return;
                }

                updateStatus(`Cloning settings from ${sourceIp} to ${destIp}...`);
                
                try {
                    const response = await fetch('/api/clone', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ source_ip: sourceIp, dest_ip: destIp })
                    });
                    
                    const result = await response.json();
                    
                    if (result.errors && result.errors.length > 0) {
                        alert('Clone complete with errors');
                    } else if (!result.success) {
                        throw new Error(result.message);
                    } else {
                        alert('Clone complete successfully.');
                    }

                    // Update data for destination IP
                    if (result.reloaded_data) {
                        cameraData[destIp] = result.reloaded_data;
                        updateStatus("Clone complete. Reloaded destination camera data.");
                    } else {
                        delete cameraData[destIp]; // Invalidate cache to force reload next time
                        updateStatus("Clone complete, but verification by reloading failed. Data may be stale.");
                    }

                    // Refresh the subgroup view to show the new settings
                    updateSubgroupSelect(subgroupSelect.value);

                } catch (e) {
                    updateStatus(`Error: ${e.message}`);
                    alert(`Error during clone: ${e.message}`);
                } finally {
                    cloneSelect.value = ""; // Reset dropdown
                }
            }

            cloneSelect.addEventListener('change', onCloneSelect);
            ipSelect.addEventListener('change', onIpSelectChange);
            mainGroupSelect.addEventListener('change', updateSubgroupSelect);
            subgroupSelect.addEventListener('change', onSubgroupChange);
            applyBtn.addEventListener('click', applySettings);
            rebootBtn.addEventListener('click', rebootCamera);
            setTimeBtn.addEventListener('click', setTime);
            reloadBtn.addEventListener('click', reloadCameraData);
            settingsView.addEventListener('input', () => applyBtn.disabled = false);
            settingsView.addEventListener('change', () => applyBtn.disabled = false);
            document.addEventListener('DOMContentLoaded', () => ipSelect.dispatchEvent(new Event('change')));
        </script></body></html>
        """

        def run_flask_app(controller, port):
            @app.route('/login', methods=['GET', 'POST'])
            def login():
                error = None
                if request.method == 'POST':
                    username, password = request.form['username'], request.form['password']
                    if simplepam.authenticate(username, password):
                        session['logged_in'] = True
                        return redirect(url_for('index'))
                    else: error = 'Invalid Credentials. Please try again.'
                return render_template_string(LOGIN_TEMPLATE, error=error)
            @app.route('/logout')
            def logout(): session.pop('logged_in', None); return redirect(url_for('login'))
            @app.route('/')
            def index():
                if not session.get('logged_in'): return redirect(url_for('login'))
                return render_template_string(MAIN_TEMPLATE, ips=controller.ips, main_groups=MAIN_GROUPS, docs=DOCS)
            @app.route('/api/data/<ip>')
            def get_data(ip):
                if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
                return jsonify(controller.load_camera_data(ip))
            @app.route('/api/apply/<ip>', methods=['POST'])
            def apply_data(ip):
                if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
                data = request.json
                result = controller.apply_settings(ip, data['subgroup'], data['settings'])
                if result['success']:
                    controller.invalidate_cache(ip)
                    reloaded_data = controller.load_camera_data(ip)
                    result['reloaded_data'] = None if 'error' in reloaded_data else reloaded_data
                return jsonify(result)
            @app.route('/api/clone', methods=['POST'])
            def clone():
                if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
                data = request.json
                source_ip = data.get('source_ip')
                dest_ip = data.get('dest_ip')
                if not source_ip or not dest_ip:
                    return jsonify({"success": False, "message": "Missing source_ip or dest_ip"}), 400
                
                clone_result = controller.clone_settings(source_ip, dest_ip)
                
                # After cloning, fetch the reloaded data for the destination IP
                # The clone_settings method already invalidated the cache
                reloaded_data = controller.load_camera_data(dest_ip)
                
                # Combine the clone result with the reloaded data
                response_data = clone_result
                response_data['reloaded_data'] = None if 'error' in reloaded_data else reloaded_data
                    
                return jsonify(response_data)
            @app.route('/api/reboot/<ip>', methods=['POST'])
            def reboot(ip):
                if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
                return jsonify(controller.reboot_camera(ip))
            @app.route('/api/set_time/<ip>', methods=['POST'])
            def set_time(ip):
                if not session.get('logged_in'): return jsonify({"error": "Unauthorized"}), 401
                return jsonify(controller.set_camera_time(ip))

            print(f"Web server running. Open http://127.0.0.1:{port} in your browser.")
            app.run(host='0.0.0.0', port=port)
        
        controller = CameraController(ips)
        run_flask_app(controller, port=args.port)

    else: # CLI mode
        if args.gop < 1: sys.exit("Error: --gop must be an integer of at least 1.")
        if args.bitrate <= 0: sys.exit("Error: --bitrate must be a positive integer.")
        print(f"Targeting cameras: {ips}")
        for ip in ips:
            cam = None
            try:
                print(f"\nConnecting to {ip}...")
                cam = DVRIPCam(ip, user='admin', password='')
                if not cam.login(): print(f"Error: Could not log in to {ip}. Skipping."); continue
                if args.dump: dump_camera_settings(cam, ip)
                else: configure_camera(cam, ip, args.codec, args.gop, args.bitrate, args.noreboot)
            except Exception as e: print(f"An unexpected error with camera {ip}: {e}")
            finally:
                if cam: cam.close(); print(f"Connection to {ip} closed.")

if __name__ == "__main__":
    main()



