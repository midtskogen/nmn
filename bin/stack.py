#!/usr/bin/env python3
import argparse
import os
import sys
import typing
import logging
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numba
import numpy as np
from tqdm import tqdm

class StackingError(Exception):
    """Custom exception for errors during the video stacking process."""
    pass

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# ==============================================================================
# Helper Functions
# ==============================================================================
def check_hw_accel() -> typing.Optional[str]:
    """
    Checks for available ffmpeg hardware acceleration methods.
    Returns "vaapi" if available, otherwise None.
    """
    try:
        result = subprocess.run(["ffmpeg", "-hwaccels"], capture_output=True, text=True, check=True)
        if "vaapi" in result.stdout:
            return "vaapi"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return None

# ==============================================================================
# Numba JIT-Optimized Functions
# ==============================================================================
@numba.jit(nopython=True, cache=True)
def jit_stack_planes(luma_stack: np.ndarray, chroma_u_stack: np.ndarray,
                     chroma_v_stack: np.ndarray, new_y: np.ndarray,
                     new_u: np.ndarray, new_v: np.ndarray) -> None:
    height, width = new_y.shape
    for i in range(height):
        for j in range(width):
            if new_y[i, j] > luma_stack[i, j]:
                luma_stack[i, j] = new_y[i, j]
                chroma_u_stack[i // 2, j // 2] = new_u[i // 2, j // 2]
                chroma_v_stack[i // 2, j // 2] = new_v[i // 2, j // 2]

@numba.jit(nopython=True, cache=True, parallel=True)
def _jit_enhance_filter_core(plane: np.ndarray, t: int, log2sizex: int, log2sizey: int) -> np.ndarray:
    height, width = plane.shape
    tmp_h = np.zeros(plane.shape, dtype=np.int16)
    final_f = np.zeros(plane.shape, dtype=np.int16)
    shiftx, shifty = 6 - log2sizex, 6 - log2sizey
    indices = np.array([-31, -23, -14, -5, 5, 14, 23, 31], dtype=np.int32)
    indices_x, indices_y = indices // (1 << shiftx), indices // (1 << shifty)

    for i in numba.prange(height):
        for j in range(width):
            center_val, h_sum = plane[i, j], 0
            for l in indices_x:
                sample_j = max(0, min(width - 1, j + l))
                sample_val = plane[i, sample_j]
                h_sum += center_val if abs(int(sample_val) - int(center_val)) > t else sample_val
            tmp_h[i, j] = h_sum

    for i in numba.prange(height):
        for j in range(width):
            center_val_h, v_sum = tmp_h[i, j], 0
            for l in indices_y:
                sample_i = max(0, min(height - 1, i + l))
                sample_val_h = tmp_h[sample_i, j]
                v_sum += center_val_h if abs(int(sample_val_h) - int(center_val_h)) > t * 4 else sample_val_h
            final_f[i, j] = v_sum
    return final_f

def enhance_filter(plane: np.ndarray, t: int, log2sizex: int, log2sizey: int,
                   dither: int, seed: int,
                   num_workers: typing.Optional[int] = None) -> np.ndarray:
    if num_workers is None: num_workers = os.cpu_count() or 1
    numba.set_num_threads(num_workers)
    log2sizex, log2sizey = np.clip(log2sizex, 3, 6), np.clip(log2sizey, 3, 6)
    final_f = _jit_enhance_filter_core(plane, t, log2sizex, log2sizey)
    c_dither = np.clip(dither, 2, 11) - 2
    dmask = (1 << c_dither) - 1
    doffset = (1 << (c_dither - 1)) - 8 if c_dither > 0 else -8
    if dmask > 0 and seed != 0:
        rng = np.random.default_rng(seed)
        noise = rng.integers(0, dmask + 1, size=plane.shape, dtype=np.int16)
        final_f += noise + doffset
    else:
        final_f += doffset
    final_f //= (1 << 6)
    return np.clip(final_f, 0, 255).astype(np.uint8)

# ==============================================================================
# Core Application Logic
# ==============================================================================
def get_video_properties_ffprobe(video_path: str) -> typing.Optional[dict]:
    if not os.path.exists(video_path):
        log.warning(f"Video file not found: '{video_path}'. Skipping.")
        return None
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration", "-of", "json", video_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)["streams"][0]
        num, den = map(int, data["r_frame_rate"].split('/'))
        data["fps"], data["duration"] = num / den, float(data.get("duration", 0))
        if not all(k in data for k in ["width", "height", "fps", "duration"]): raise ValueError("Missing essential video stream data.")
        return data
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, ValueError, KeyError) as e:
        log.error(f"Could not read valid properties from '{video_path}' using ffprobe: {e}")
        return None

def video_stack_worker(task: dict, resize_factor: float = 1.0, use_hw_accel: bool = False):
    video_path = task["path"]
    props = get_video_properties_ffprobe(video_path)
    if not props: return None

    width, height = props["width"], props["height"]
    if resize_factor != 1.0:
        width, height = int(width * resize_factor), int(height * resize_factor)
        width, height = (width // 2) * 2, (height // 2) * 2

    luma_stack = np.zeros((height, width), dtype=np.uint8)
    chroma_u_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    chroma_v_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)

    ffmpeg_command = ["ffmpeg", "-loglevel", "warning"]
    video_filters, output_pix_fmt = [], "yuv420p"

    if use_hw_accel:
        ffmpeg_command.extend(["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"])
        if resize_factor != 1.0: video_filters.append(f"scale_vaapi=w={width}:h={height}")
        video_filters.extend(["hwdownload,format=nv12", f"format={output_pix_fmt}"])
    else:
        if resize_factor != 1.0: video_filters.append(f"scale=w={width}:h={height}")

    ffmpeg_command.extend(["-i", video_path, "-ss", str(task["start_seconds"]), "-t", str(task["duration_seconds"])])
    if video_filters: ffmpeg_command.extend(["-vf", ",".join(video_filters)])
    ffmpeg_command.extend(["-f", "rawvideo", "-pix_fmt", output_pix_fmt, "pipe:1"])
    
    proc = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    y_size, uv_size = width * height, (width // 2) * (height // 2)
    frame_size = y_size + uv_size * 2
    
    while True:
        frame_data = proc.stdout.read(frame_size)
        if len(frame_data) < frame_size:
            break
        
        y_plane = np.frombuffer(frame_data, dtype=np.uint8, count=y_size, offset=0).reshape((height, width))
        u_plane = np.frombuffer(frame_data, dtype=np.uint8, count=uv_size, offset=y_size).reshape((height // 2, width // 2))
        v_plane = np.frombuffer(frame_data, dtype=np.uint8, count=uv_size, offset=y_size + uv_size).reshape((height // 2, width // 2))
        jit_stack_planes(luma_stack, chroma_u_stack, chroma_v_stack, y_plane, u_plane, v_plane)

    return_code = proc.wait()
    if return_code != 0:
        stderr_output = proc.stderr.read().decode('utf-8')
        log.error(f"ffmpeg process for {video_path} failed with exit code {return_code}:\n{stderr_output}")

    proc.stdout.close(); proc.stderr.close()
    
    return luma_stack, chroma_u_stack, chroma_v_stack, task

def stack_video_frames(video_paths: list, output_path: str,
                       start_seconds: float, duration_seconds: typing.Optional[float],
                       denoise: bool, denoise_strength: float, quality: int,
                       num_threads: int, individual: bool, enhance: bool,
                       progress: bool = False, resize_factor: float = 1.0,
                       use_hw_accel: bool = False):
    processing_plan = []
    log.info("Pre-scanning video files to create processing plan...")
    time_cursor = 0.0
    for path in video_paths:
        props = get_video_properties_ffprobe(path)
        if not props: continue
        video_duration = props["duration"]
        end_boundary = float('inf') if duration_seconds is None else start_seconds + duration_seconds
        effective_start_time, effective_end_time = max(time_cursor, start_seconds), min(time_cursor + video_duration, end_boundary)
        if effective_end_time > effective_start_time:
            seek_start_in_file, process_duration = effective_start_time - time_cursor, effective_end_time - effective_start_time
            if process_duration > 0:
                processing_plan.append({"path": path, "start_seconds": seek_start_in_file, "duration_seconds": process_duration})
        time_cursor += video_duration

    if not processing_plan: raise StackingError("No video frames found to process with the given time settings.")
    
    props = get_video_properties_ffprobe(video_paths[0])
    width, height = props['width'], props['height']
    if resize_factor != 1.0:
        width, height = int(width * resize_factor), int(height * resize_factor)
        width, height = (width // 2) * 2, (height // 2) * 2

    log.info(f"Using software decoding by default. Ready to process {len(processing_plan)} video segments using up to {num_threads} threads.")
    total_luma, total_u, total_v = np.zeros((height, width), dtype=np.uint8), np.full((height // 2, width // 2), 128, dtype=np.uint8), np.full((height // 2, width // 2), 128, dtype=np.uint8)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(video_stack_worker, task, resize_factor, use_hw_accel) for task in processing_plan]
        pbar_context = tqdm(total=len(processing_plan), desc="Processing Segments", unit="seg", disable=not progress)
        with pbar_context as pbar:
            for future in futures:
                result = future.result()
                pbar.update(1)
                if result is None: continue
                luma, u, v, task = result
                jit_stack_planes(total_luma, total_u, total_v, luma, u, v)
                if individual:
                    base_name = os.path.splitext(os.path.basename(task['path']))[0]
                    individual_output_path = f"{base_name}_max.jpg"
                    finalize_and_save(luma, u, v, individual_output_path, denoise, denoise_strength, quality, enhance, num_threads, progress=progress)

    if np.all(total_luma == 0): raise StackingError("Stacking resulted in a black image. No frames were processed.")
    
    if not individual:
        log.info("Finalizing combined stack of all videos...")
        finalize_and_save(total_luma, total_u, total_v, output_path, denoise, denoise_strength, quality, enhance, num_threads, progress=progress)

def finalize_and_save(luma_plane: np.ndarray, u_plane: np.ndarray,
                      v_plane: np.ndarray, output_path: str, denoise: bool,
                      denoise_strength: float, quality: int, enhance: bool,
                      num_threads: int, progress: bool = False):
    if enhance:
        log.info(f"Applying enhancement filter to '{output_path}'...")
        seed_y = int.from_bytes(os.urandom(4), 'little')
        luma_plane = enhance_filter(luma_plane, t=12, log2sizex=5, log2sizey=5, dither=6, seed=seed_y, num_workers=num_threads)
        u_plane = enhance_filter(u_plane, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0, num_workers=num_threads)
        v_plane = enhance_filter(v_plane, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0, num_workers=num_threads)

    height, width = luma_plane.shape
    yuv_i420 = np.vstack([ luma_plane, u_plane.reshape((height // 4, width)), v_plane.reshape((height // 4, width)) ])
    bgr_image = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
    
    if denoise:
        log.info(f"Denoising '{output_path}' with strength {denoise_strength}...")
        bgr_image = cv2.fastNlMeansDenoisingColored(
            bgr_image, None, h=denoise_strength, hColor=denoise_strength,
            templateWindowSize=7, searchWindowSize=21)
    
    log.info(f"Saving result to '{output_path}'...")
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(output_path, bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(output_path, bgr_image)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Stack video frames to simulate a long exposure.", formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("video_files", nargs='*', help="Path(s) to input video file(s).")
    parser.add_argument("-o", "--output", default="max.jpg", help="Path for the final output image.")
    parser.add_argument("-s", "--start", type=float, default=0.0, help="Start position in seconds.")
    parser.add_argument("-t", "--duration", type=float, default=None, help="Total duration in seconds to stack from the start time.")
    parser.add_argument("--individual", action="store_true", help="Save a separate stacked image for each input file.")
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101), metavar="{1-100}", help="Quality for JPEG output (1-100, default: 95).")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="Number of threads for parallel processing (default: all cores).")
    parser.add_argument("--resize", type=float, default=1.0, help="Factor to resize video frames (e.g., 0.5 for 50% size). Default: 1.0 (no resize).")
    parser.add_argument("--force-hwaccel", action="store_true", help="Force use of VA-API hardware acceleration (may be slower).")

    noise_group = parser.add_argument_group('Noise Reduction')
    noise_mutex = noise_group.add_mutually_exclusive_group()
    noise_mutex.add_argument("--denoise", action="store_true", help="Use OpenCV's denoising filter instead of the default enhancement filter.")
    noise_mutex.add_argument("--noenhance", action="store_true", help="Disable the default enhancement filter. No noise reduction will be applied.")
    noise_group.add_argument("--denoise-strength", type=float, default=10.0, help="Strength of the --denoise filter (default: 10.0).")
    
    args = parser.parse_args()
    enhance = not (args.noenhance or args.denoise)
    
    use_hw_accel = False
    if args.force_hwaccel:
        if check_hw_accel() == "vaapi":
            log.info("✅ Forcing VA-API hardware acceleration as requested.")
            use_hw_accel = True
        else:
            log.warning("⚠️ --force-hwaccel specified, but 'vaapi' is not available. Falling back to software decoding.")

    if not args.video_files: parser.error("Missing input. Provide at least one video file.")
        
    try:
        stack_video_frames(
            args.video_files, args.output, args.start, args.duration,
            args.denoise, args.denoise_strength, args.quality,
            args.threads, args.individual, enhance, progress=True,
            resize_factor=args.resize, use_hw_accel=use_hw_accel
        )
    except StackingError as e:
        log.error(f"A fatal error occurred: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        log.error(f"A required command is not found. Is ffmpeg/ffprobe in your PATH? Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
