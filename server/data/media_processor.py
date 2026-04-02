#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# --- Try to import from pto_mapper.py ---
# This attempts to import the necessary functions for camera calibration transformations.
# The PTO_MAPPER_AVAILABLE flag is used by other modules to conditionally enable features
# that rely on this functionality.
try:
    from pto_mapper import map_image_to_pano, map_pano_to_image
    PTO_MAPPER_AVAILABLE = True
except ImportError:
    PTO_MAPPER_AVAILABLE = False

# --- Airline Code Lookup ---
_airline_codes = None

def get_airline_codes():
    """Loads and caches airline ICAO-to-IATA code mappings from a JSON file."""
    global _airline_codes
    if _airline_codes is not None:
        return _airline_codes

    # The controller.py script runs from the 'cgi-bin' directory.
    # We assume airline_codes.json is in that same directory.
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        codes_file_path = os.path.join(base_dir, 'airline_codes.json')
        with open(codes_file_path, 'r') as f:
            data = json.load(f)
        # Create a mapping from ICAO -> {iata, name}, filtering out entries without an IATA code.
        _airline_codes = {k: {"iata": v["IATA"], "name": v["Name"]} for k, v in data.items() if v.get("IATA")}
    except (IOError, json.JSONDecodeError):
        logging.warning("airline_codes.json not found or is invalid. Falling back to ICAO callsigns for overlays.")
        _airline_codes = {} # Cache empty dict to prevent reload attempts
    return _airline_codes


def get_sun_altitude(dt_utc, lat, lon):
    """
    Calculates the sun's altitude for a given UTC datetime, latitude, and longitude.
    This is used to determine day/twilight/night conditions for styling the track overlays.
    """
    day_of_year = dt_utc.timetuple().tm_yday
    lat_rad = math.radians(lat)
    
    # Simplified calculation for Equation of Time and solar declination.
    B = math.radians((360/365) * (day_of_year - 81))
    eot = 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    solar_declination = math.radians(-23.44 * math.cos(math.radians((360/365) * (day_of_year + 10))))
    
    # Calculate the hour angle based on the true solar time.
    time_offset = eot + 4 * lon
    true_solar_time_minutes = dt_utc.hour * 60 + dt_utc.minute + dt_utc.second / 60 + time_offset
    hour_angle = (true_solar_time_minutes / 4) - 180
    hour_angle_rad = math.radians(hour_angle)
    
    # Final formula to calculate the sine of the sun's altitude.
    sin_altitude = math.sin(lat_rad) * math.sin(solar_declination) + math.cos(lat_rad) * math.cos(solar_declination) * math.cos(hour_angle_rad)
    return math.degrees(math.asin(sin_altitude))


def stack_images(image_paths, output_path, task_id):
    """
    Stacks a list of images into a single image using a maximum intensity projection.
    This is equivalent to a "lighten" blend mode in photo editing software and is used
    to create long-exposure style star trail or meteor shower composites.
    """
    if not image_paths:
        logging.warning(f"Task {task_id} - stack_images called with no images.")
        return False
    try:
        logging.info(f"Task {task_id} - Stacking {len(image_paths)} images into {os.path.basename(output_path)}.")
        
        stacked_np = None
        
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert('RGB')
                    img_np = np.array(img_rgb)
                    
                    if stacked_np is None:
                        # Initialize the stacked image array with the first image.
                        stacked_np = img_np
                    else:
                        # Ensure subsequent images have the same dimensions.
                        if stacked_np.shape != img_np.shape:
                            logging.warning(f"Task {task_id} - Mismatched image dimensions, skipping {path}. Expected {stacked_np.shape}, got {img_np.shape}.")
                            continue
                        # The core stacking logic: take the maximum (brightest) pixel value from either image.
                        stacked_np = np.maximum(stacked_np, img_np)
            except Exception as e:
                logging.error(f"Task {task_id} - Could not open or process image {path} for stacking: {e}")
                continue
        
        # Save the final stacked numpy array as a JPEG image.
        if stacked_np is not None:
            final_image = Image.fromarray(stacked_np)
            final_image.save(output_path, 'jpeg', quality=95)
            logging.info(f"Task {task_id} - Successfully saved stacked image to {output_path}.")
            return True
        else:
            logging.error(f"Task {task_id} - Failed to stack any images (all inputs may have failed).")
            return False

    except Exception as e:
        logging.error(f"Task {task_id} - A critical error occurred during image stacking: {e}", exc_info=True)
        return False


def draw_track_on_image(pto_data, pass_info, output_path, target_w=None, target_h=None, is_flight=False):
    """
    Draws a satellite or flight track onto a transparent PNG image.
    This PNG can then be used as an overlay on top of a video or still image.
    It uses pto_mapper to convert astronomical coordinates (azimuth, altitude) to pixel coordinates.
    """
    if not PTO_MAPPER_AVAILABLE:
        raise RuntimeError("Cannot draw track on image: pto_mapper library is not available.")
    
    sky_track = pass_info.get('sky_track', [])
    if not sky_track: raise ValueError("Pass information is missing sky_track data.")
    _, images = pto_data
    if not images: raise ValueError("PTO data contains no image information.")
    
    img_info = images[0]
    pto_w, pto_h = img_info.get('w'), img_info.get('h')
    if not pto_w or not pto_h: raise ValueError("Image dimensions 'w' and 'h' not found in PTO data.")
    
    # Determine the final dimensions of the overlay image.
    final_w = int(target_w) if target_w is not None else int(pto_w)
    final_h = int(target_h) if target_h is not None else int(pto_h)
    scale_x = final_w / pto_w
    scale_y = final_h / pto_h
    
    # Create a new transparent image to draw on.
    img = Image.new('RGBA', (final_w, final_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    line_points, time_points, altitude_points, opacity_points = [], [], [], []
    # Extend the track slightly to help with Catmull-Rom spline calculations at the ends.
    extended_sky_track = [sky_track[0]] + sky_track + [sky_track[-1]] if len(sky_track) > 1 else sky_track
    
    # Convert all sky track (az, alt) points to pixel coordinates.
    for point in extended_sky_track:
        pano_x, pano_y = point['az'] * 100, (90 - point['alt']) * 100
        img_coords = map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=False)
        if img_coords:
            _, sx, sy = img_coords
            scaled_point = (sx * scale_x, sy * scale_y)
            # Add point only if it has moved to avoid clutter.
            if not line_points or (abs(scaled_point[0] - line_points[-1][0]) > 0.1 or abs(scaled_point[1] - line_points[-1][1]) > 0.1):
                line_points.append(scaled_point)
                time_points.append(point['time'])
                altitude_points.append(point['alt'])
                opacity_points.append(point.get('opacity'))

    if len(line_points) > 1:
        line_width = max(1, int(final_w / 128.0))
        # Add padding for spline calculation.
        padded_points = [line_points[0]] + line_points + [line_points[-1]]

        # Draw the track segment by segment using Catmull-Rom splines for a smooth curve.
        for i in range(1, len(padded_points) - 2):
            p0, p1, p2, p3 = (np.array(p) for p in [padded_points[i-1], padded_points[i], padded_points[i+1], padded_points[i+2]])
            alpha = 0
            base_color_rgb = (255, 255, 255) # Track color is white for both satellites and flights.

            # Determine the transparency of the track segment.
            if is_flight:
                # For flights, use the pre-calculated distance-based opacity.
                opacity_float = opacity_points[i-1] if opacity_points[i-1] is not None else 0.5
                alpha = int(opacity_float * 255)
            else: 
                # For satellites, transparency is based on altitude in the sky.
                avg_alt = (altitude_points[i-1] + altitude_points[i]) / 2.0
                transparency_percent = 5.0 + max(0, 15.0 - (90.0 - avg_alt) / 3.5)
                alpha = int(128 * (transparency_percent / 100.0))

            track_color = (*base_color_rgb, alpha)
            
            # Calculate points along the Catmull-Rom spline segment.
            segment_curve_points = []
            for t_int in range(17):
                t = t_int / 16
                point = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)
                segment_curve_points.append(tuple(point))
            draw.line(segment_curve_points, fill=track_color, width=line_width, joint="curve")

        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except IOError: font = ImageFont.load_default()
            
        last_drawn_time = None
        # Draw timestamp labels along the track.
        for i in range(1, len(line_points) - 1):
            dt = datetime.fromisoformat(time_points[i].replace('Z', '+00:00'))
            # For satellites, thin out labels at lower altitudes to reduce clutter.
            if not is_flight and altitude_points[i] < 45 and last_drawn_time and (dt - last_drawn_time).total_seconds() < 14.5:
                continue
            
            p1, p2, (px, py) = line_points[i-1], line_points[i+1], line_points[i]
            # Calculate the local angle of the track to rotate the text label.
            track_angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            text_angle_deg = 90 - math.degrees(track_angle_rad)
            side_multiplier = 1.0 # To place text on one side of the line.
            if text_angle_deg > 90:
                text_angle_deg -= 180
                side_multiplier = -1.0
            elif text_angle_deg < -90:
                text_angle_deg += 180
                side_multiplier = -1.0
          
            time_str = "    " + dt.strftime("%H:%M:%S")
            try:
                bbox = font.getbbox(time_str)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(time_str, font=font)
                
            # Draw the rotated text on a temporary canvas.
            canvas_size = int(math.sqrt(text_w**2 + text_h**2)) * 2
            txt_canvas = Image.new('RGBA', (canvas_size, canvas_size), (0,0,0,0))
            txt_draw = ImageDraw.Draw(txt_canvas)
            txt_draw.text((canvas_size/2, canvas_size/2), time_str, font=font, fill=(255, 255, 0, 242), anchor="lm")
            
            rotated_txt = txt_canvas.rotate(text_angle_deg, resample=Image.BICUBIC, center=(canvas_size/2, canvas_size/2))
            # Position the text slightly offset from the track line.
            offset_dist = (line_width / 2 + 2) * side_multiplier
            target_x = px - offset_dist * math.sin(track_angle_rad)
            target_y = py + offset_dist * math.cos(track_angle_rad)
            
            # Paste the rotated text onto the main overlay image.
            img.paste(rotated_txt, (int(target_x - canvas_size/2), int(target_y - canvas_size/2)), rotated_txt)
            if not is_flight: last_drawn_time = dt
    
    # If it's a flight, add the callsign to the corner of the image.
    if is_flight:
        callsign_raw = pass_info.get('flight_info', {}).get('callsign', '').strip()
        label_text = callsign_raw # Default to the original callsign

        if callsign_raw and len(callsign_raw) > 3:
            icao_code = callsign_raw[:3].upper()
            flight_num = callsign_raw[3:]
            
            airline_codes = get_airline_codes()
            airline_info = airline_codes.get(icao_code)
            
            if airline_info:
                label_text = f"{airline_info['iata']}{flight_num}" # Use IATA + flight number
        
        if label_text:
            try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except IOError: font = ImageFont.load_default()
            
            margin = 10
            try:
                bbox = font.getbbox(label_text)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(label_text, font=font)
            
            x_pos = final_w - text_w - margin
            y_pos = final_h - text_h - margin
            
            draw.rectangle((x_pos - 5, y_pos - 5, x_pos + text_w + 5, y_pos + text_h + 5), fill=(0, 0, 0, 128))
            draw.text((x_pos, y_pos), label_text, font=font, fill=(255, 255, 255, 255))

    img.save(output_path, 'PNG')


def apply_ffmpeg_overlay(base_media_path, overlay_path, output_path):
    """
    Uses ffmpeg to apply a transparent PNG overlay onto a video or image file.
    This is faster than using Python libraries for video processing.
    """
    base, ext = os.path.splitext(output_path)
    temp_output_path = f"{base}.tmp{ext}"
    try:
        is_video = base_media_path.lower().endswith('.mp4')
        is_hevc_output = output_path.lower().endswith('_hevc.mp4')
        
        # Select appropriate video codec options based on the output format.
        video_codec_opts = []
        if is_video:
            if is_hevc_output: video_codec_opts = ["-c:v", "libx265", "-preset", "veryfast", "-tag:v", "hvc1"]
            else: video_codec_opts = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]

        audio_codec_opts = ["-c:a", "copy"] if is_video else []
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", base_media_path,
            "-i", overlay_path, "-filter_complex", "[0:v][1:v]overlay",
            *video_codec_opts, *audio_codec_opts, "-y", temp_output_path
        ]
        logging.info(f"Executing ffmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        os.rename(temp_output_path, output_path)
        logging.info(f"Successfully applied overlay to {os.path.basename(output_path)}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg overlay FAILED for {os.path.basename(base_media_path)}.")
        logging.error(f"ffmpeg stderr: {e.stderr.decode('utf-8', errors='ignore')}")
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False
    except Exception as e:
        logging.error(f"Unexpected Python error during ffmpeg call for {os.path.basename(base_media_path)}: {e}")
        if os.path.exists(temp_output_path): os.remove(temp_output_path)
        return False


def create_thumbnail(task_id, path, file_type, station_code, cam_num, has_overlay=False, is_flight=False, max_file_size_mb=200, flight_track_text="(flight track)", sat_track_text="(satellite track)"):
    """
    Creates a JPG thumbnail for a given video or image file.
    It adds a text overlay indicating the station/camera and if a track is present.
    """
    thumb_path = None
    try:
        # Skips thumbnail creation for excessively large files to save resources.
        if not os.path.exists(path) or os.path.getsize(path) > (max_file_size_mb * 1024 * 1024):
            return None
        
        thumb_path = f"{os.path.splitext(path)[0]}_thumb.jpg"
        
        if file_type.startswith('image'):
            # For images, use Pillow for resizing.
            with Image.open(path) as img:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img = img.convert('RGB') if img.mode != 'RGB' else img
                img.save(thumb_path, "jpeg", quality=85)
        else:
            # For videos, use ffmpeg to extract the first frame, which is much faster.
            subprocess.run(["ffmpeg", "-i", path, "-ss", "00:00:01", "-vframes", "1", "-vf", "scale=512:-1", "-y", thumb_path], check=True, capture_output=True)
        
        # Add text labels to the generated thumbnail.
        if os.path.exists(thumb_path):
            with Image.open(thumb_path) as thumb_img:
                thumb_img = thumb_img.convert('RGBA') if thumb_img.mode != 'RGBA' else thumb_img
                txt_overlay = Image.new('RGBA', thumb_img.size, (255,255,255,0))
                draw = ImageDraw.Draw(txt_overlay)
                try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
                except IOError: font = ImageFont.load_default()
       
                # Draw station and camera label with a semi-transparent background.
                station_text = f"{station_code}:{cam_num}"
                try:
                    bbox = font.getbbox(station_text)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    w, h = draw.textsize(station_text, font=font)
                
                img_w, img_h = thumb_img.size
                sx, sy = (img_w - w) / 2, 10
           
                draw.rectangle((sx - 5, sy - 5, sx + w + 5, sy + h + 5), fill=(0, 0, 0, 128))
                draw.text((sx, sy), station_text, font=font, fill="white")

                # If the image has an overlay, add a label indicating it.
                if has_overlay:
                    track_text = flight_track_text if is_flight else sat_track_text
                    try:
                        bbox = font.getbbox(track_text)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        w, h = draw.textsize(track_text, font=font)
                    tx, ty = (img_w - w) / 2, img_h - h - 10
                    draw.rectangle((tx - 5, ty - 5, tx + w + 5, ty + h + 5), fill=(0, 0, 0, 128))
                    draw.text((tx, ty), track_text, font=font, fill="white")
                
                # Composite the text overlay onto the thumbnail and save as final JPG.
                thumb_img = Image.alpha_composite(thumb_img, txt_overlay).convert('RGB')
                thumb_img.save(thumb_path, "jpeg", quality=90)

        return os.path.basename(thumb_path)
    except Exception as e:
        logging.error(f"Task {task_id} - Thumbnail creation failed for {path}: {e}")
        if thumb_path and os.path.exists(thumb_path):
            try: os.remove(thumb_path)
            except OSError: pass
        return None


def probe_video_codec(filepath, task_id):
    """
    Probes a video file using ffprobe to determine its video codec.
    Returns the codec name as a lowercase string (e.g., 'hevc', 'h264').
    """
    try:
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        codec = result.stdout.strip().lower()
        logging.info(f"Task {task_id} - Probed {os.path.basename(filepath)}: detected codec '{codec}'.")
        return codec
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logging.error(f"Task {task_id} - ffprobe failed for {os.path.basename(filepath)}: {e.stderr if hasattr(e, 'stderr') else e}")
        return 'h264' # Assume h264 on failure as a safe fallback.
