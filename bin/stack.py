#!/usr/bin/env python3
import argparse
import os
import sys
import typing
from concurrent.futures import ThreadPoolExecutor

import cv2
import numba
import numpy as np
from tqdm import tqdm

# ==============================================================================
# Numba JIT-Optimized Functions
# ==============================================================================
# These functions contain computationally-heavy, looping logic that is
# compiled to fast machine code by Numba for significant performance gains.

@numba.jit(nopython=True, cache=True)
def jit_stack_planes(luma_stack: np.ndarray, chroma_u_stack: np.ndarray,
                     chroma_v_stack: np.ndarray, new_y: np.ndarray,
                     new_u: np.ndarray, new_v: np.ndarray) -> None:
    """
    Updates the Y, U, and V planes with data from a new frame.

    This function finds all pixels where the new luma plane (new_y) is
    brighter than the existing stack and updates the luma, U, and V stacks
    at those locations. The U and V planes are assumed to be half the
    resolution of the luma plane (YUV420 format).

    Args:
        luma_stack: The existing stacked luma (Y) plane. Modified in-place.
        chroma_u_stack: The existing stacked chroma (U) plane. Modified in-place.
        chroma_v_stack: The existing stacked chroma (V) plane. Modified in-place.
        new_y: The new luma plane to compare against.
        new_u: The new chroma (U) plane.
        new_v: The new chroma (V) plane.
    """
    height, width = new_y.shape
    for i in range(height):
        for j in range(width):
            if new_y[i, j] > luma_stack[i, j]:
                luma_stack[i, j] = new_y[i, j]
                # Update corresponding pixel in half-resolution chroma planes
                chroma_u_stack[i // 2, j // 2] = new_u[i // 2, j // 2]
                chroma_v_stack[i // 2, j // 2] = new_v[i // 2, j // 2]


def enhance_filter(plane: np.ndarray, t: int, log2sizex: int, log2sizey: int,
                   dither: int, seed: int) -> np.ndarray:
    """
    Replicates the C-style adaptive enhancement/denoise filter using numpy vectorization.

    This version is designed for performance using NumPy's optimized array operations
    and is easily importable into other scripts.

    Args:
        plane: The 2D numpy array (image plane) to filter.
        t: The adaptive threshold for pixel comparison.
        log2sizex: The log2 of the horizontal filter size.
        log2sizey: The log2 of the vertical filter size.
        dither: The dithering amount.
        seed: The seed for the random number generator.

    Returns:
        The filtered 8-bit image plane.
    """
    height, width = plane.shape
    plane_f = plane.astype(np.int32)

    # --- Parameter setup ---
    log2sizex = np.clip(log2sizex, 3, 6)
    log2sizey = np.clip(log2sizey, 3, 6)
    sizex = 1 << log2sizex
    sizey = 1 << log2sizey
    size2x = sizex >> 1
    size2y = sizey >> 1
    shiftx = 6 - log2sizex
    shifty = 6 - log2sizey
    
    indices = np.array([-31, -23, -14, -5, 5, 14, 23, 31], dtype=np.int32)
    log2indices = 3

    # --- Horizontal Pass ---
    indices_x = (indices // (1 << shiftx))
    # Pad the image horizontally to handle edges during sampling
    padded_h = np.pad(plane_f, ((0, 0), (size2x, size2x)), 'edge')
    tmp_h = np.zeros_like(plane_f)
    for l in indices_x:
        # Create a view of the sampled pixels, shifted by 'l'
        sample = padded_h[:, size2x + l : size2x + l + width]
        # Apply the adaptive threshold: if a sample is too different, use the center pixel's value instead
        filtered_sample = np.where(np.abs(sample - plane_f) > t, plane_f, sample)
        tmp_h += filtered_sample

    # --- Vertical Pass ---
    indices_y = (indices // (1 << shifty))
    # Pad the result of the horizontal pass vertically
    padded_v = np.pad(tmp_h, ((size2y, size2y), (0, 0)), 'edge')
    final_f = np.zeros_like(plane_f)
    for l in indices_y:
        sample = padded_v[size2y + l : size2y + l + height, :]
        # Apply a larger threshold for the vertical pass
        filtered_sample = np.where(np.abs(sample - tmp_h) > t * 4, tmp_h, sample)
        final_f += filtered_sample
    
    # --- Dithering ---
    c_dither = dither
    if c_dither < 2: c_dither = 2
    c_dither -= 2
    if c_dither > 11: c_dither = 11
    
    c_seed = seed if seed != 0 else 1
    dmask = (1 << c_dither) - 1
    doffset = (1 << (c_dither - 1)) - 8 if c_dither > 0 else -8
    
    if dmask > 0 and seed != 0:
        # Use NumPy's modern random number generator for better performance and reproducibility
        rng = np.random.default_rng(c_seed)
        noise = rng.integers(0, dmask + 1, size=plane.shape, dtype=np.int32) + doffset
        final_f += noise
    else:
        final_f += doffset

    # --- Scale down and clip to final 8-bit range ---
    scale_factor = 1 << (2 * log2indices)
    final_f //= scale_factor
    
    return np.clip(final_f, 0, 255).astype(np.uint8)


# ==============================================================================
# Core Application Logic
# ==============================================================================

def get_video_properties(
    video_path: str
) -> typing.Tuple[typing.Optional[cv2.VideoCapture], ...]:
    """Safely opens a video and returns its properties."""
    if not os.path.exists(video_path):
        tqdm.write(f"Warning: Video file not found: '{video_path}'. Skipping.",
                   file=sys.stderr)
        return None, None, None, None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(
            f"Warning: Could not open video file: '{video_path}'. Skipping.",
            file=sys.stderr)
        return None, None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not all([width, height, fps > 0, frame_count > 0]):
        tqdm.write(
            f"Warning: Could not read valid properties from '{video_path}'. Skipping.",
            file=sys.stderr)
        cap.release()
        return None, None, None, None, None

    return cap, width, height, fps, frame_count


def video_stack_worker(
    task: dict
) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """
    A worker function that creates a stacked image for a single video file.

    This function is designed to be run in a separate thread. It reads frames
    from a video, converts them to YUV420, and uses the JIT-optimized
    `jit_stack_planes` function to perform the stacking.

    Returns:
        A tuple containing the Y, U, V stacked planes and the original task.
    """
    cap, width, height, _, _ = get_video_properties(task["path"])
    if not cap: return None

    # Initialize YUV420 planes for stacking
    luma_stack = np.zeros((height, width), dtype=np.uint8)
    chroma_u_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    chroma_v_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)

    cap.set(cv2.CAP_PROP_POS_FRAMES, task["start_frame"])

    for _ in range(task["frame_count"]):
        ret, frame = cap.read()
        if not ret: break

        # Convert BGR frame to YUV I420 (planar) format
        yuv_i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        
        # Extract planes
        y_plane = yuv_i420[0:height, :]
        uv_plane_end = height + (height // 4)
        u_plane = yuv_i420[height:uv_plane_end, :].reshape((height // 2, width // 2))
        v_plane = yuv_i420[uv_plane_end:, :].reshape((height // 2, width // 2))

        # Update the stacks using the fast JIT function
        jit_stack_planes(luma_stack, chroma_u_stack, chroma_v_stack,
                         y_plane, u_plane, v_plane)

    cap.release()
    return luma_stack, chroma_u_stack, chroma_v_stack, task


def stack_video_frames(video_paths: list, output_path: str,
                       start_seconds: float,
                       duration_seconds: typing.Optional[float],
                       denoise: bool, denoise_strength: float, quality: int,
                       num_threads: int, individual: bool, enhance: bool):
    """
    Reads and stacks video files in parallel.
    """
    processing_plan = []
    
    # --- Planning Phase ---
    print("Pre-scanning video files to create processing plan...")
    time_cursor = 0.0
    main_cap, width, height, _, _ = get_video_properties(video_paths[0])
    if not main_cap: sys.exit("Error: Could not open the first video file.")
    main_cap.release()
    
    for path in video_paths:
        cap, _, _, fps, frame_count = get_video_properties(path)
        if not cap: continue
        video_duration = frame_count / fps
        end_boundary = float('inf') if duration_seconds is None else start_seconds + duration_seconds
        effective_start_time = max(time_cursor, start_seconds)
        effective_end_time = min(time_cursor + video_duration, end_boundary)

        if effective_end_time > effective_start_time:
            start_frame = int((effective_start_time - time_cursor) * fps)
            frames_to_process = int((effective_end_time - effective_start_time) * fps)
            if frames_to_process > 0:
                processing_plan.append({
                    "path": path, "start_frame": start_frame, "frame_count": frames_to_process
                })
        time_cursor += video_duration
        cap.release()

    if not processing_plan:
        print("Error: No frames found to process with the given settings.", file=sys.stderr)
        sys.exit(1)

    # --- Execution Phase ---
    print(f"Ready to process {len(processing_plan)} video file(s) using up to {num_threads} threads.")
    total_luma = np.zeros((height, width), dtype=np.uint8)
    total_u = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    total_v = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(video_stack_worker, task) for task in processing_plan]
        
        with tqdm(total=len(processing_plan), desc="Processing Files", unit="file") as pbar:
            for future in futures:
                result = future.result()
                pbar.update(1)
                if result is None: continue
                
                luma, u, v, task = result
                
                # Stack results from this video onto the total stack
                jit_stack_planes(total_luma, total_u, total_v, luma, u, v)

                if individual:
                    base_name = os.path.splitext(os.path.basename(task['path']))[0]
                    individual_output_path = f"{base_name}_max.jpg"
                    finalize_and_save(luma, u, v, individual_output_path,
                                      denoise, denoise_strength, quality,
                                      enhance, pbar=pbar)

    if np.all(total_luma == 0):
        print("\nNo frames were stacked. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print("\nFinalizing combined stack of all videos...")
    finalize_and_save(total_luma, total_u, total_v, output_path, denoise,
                      denoise_strength, quality, enhance)


def stack_raw_frames(width: int, height: int, output_path: str,
                     start_seconds: float,
                     duration_seconds: typing.Optional[float], fps: float,
                     denoise: bool, denoise_strength: float, quality: int,
                     enhance: bool):
    """
    Reads raw YUV420 data from stdin, stacks frames, and saves the result.
    """
    print(f"Processing raw YUV420 stream at {width}x{height}, {fps} FPS.")
    
    y_size, uv_size = width * height, (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    luma_stack = np.zeros((height, width), dtype=np.uint8)
    chroma_u_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    chroma_v_stack = np.full((height // 2, width // 2), 128, dtype=np.uint8)

    frames_to_skip = int(start_seconds * fps)
    if frames_to_skip > 0:
        print(f"Skipping {frames_to_skip} frames ({start_seconds}s)...")
        # Read in chunks to avoid multiple small reads
        bytes_to_skip = frames_to_skip * frame_size
        while bytes_to_skip > 0:
            chunk_size = min(bytes_to_skip, 65536)
            if not sys.stdin.buffer.read(chunk_size): break
            bytes_to_skip -= chunk_size
    
    total_frames = int(duration_seconds * fps) if duration_seconds is not None else None
    
    with tqdm(total=total_frames, desc="Stacking Raw Frames", unit="frame") as pbar:
        while True:
            if total_frames is not None and pbar.n >= total_frames: break
            frame_data = sys.stdin.buffer.read(frame_size)
            if len(frame_data) < frame_size: break

            y_plane = np.frombuffer(frame_data[:y_size], dtype=np.uint8).reshape((height, width))
            u_plane = np.frombuffer(frame_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((height//2, width//2))
            v_plane = np.frombuffer(frame_data[y_size + uv_size:], dtype=np.uint8).reshape((height//2, width//2))
            
            jit_stack_planes(luma_stack, chroma_u_stack, chroma_v_stack, y_plane, u_plane, v_plane)
            pbar.update(1)

    if pbar.n == 0:
        print("\nNo frames were stacked from raw input. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print("\nReconstructing and saving final image...")
    finalize_and_save(luma_stack, chroma_u_stack, chroma_v_stack,
                      output_path, denoise, denoise_strength, quality, enhance,
                      pbar=pbar)


def finalize_and_save(luma_plane: np.ndarray, u_plane: np.ndarray,
                      v_plane: np.ndarray, output_path: str, denoise: bool,
                      denoise_strength: float, quality: int, enhance: bool,
                      pbar: typing.Optional[tqdm] = None):
    """
    Applies final filters, converts YUV planes to BGR, and saves the image.
    """
    log = tqdm.write if pbar else print
    
    if enhance:
        log(f"Applying enhancement filter to '{output_path}'...")
        seed_y = int.from_bytes(os.urandom(4), 'little')
        luma_plane = enhance_filter(luma_plane, t=12, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
        u_plane = enhance_filter(u_plane, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
        v_plane = enhance_filter(v_plane, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

    # Reconstruct the final BGR image from YUV420 planes
    height, width = luma_plane.shape
    yuv_i420 = np.vstack([
        luma_plane,
        u_plane.reshape((height // 4, width)),
        v_plane.reshape((height // 4, width))
    ])
    bgr_image = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)
    
    final_image = bgr_image
    if denoise:
        log(f"Denoising '{output_path}' with strength {denoise_strength}...")
        final_image = cv2.fastNlMeansDenoisingColored(
            bgr_image, None, h=denoise_strength, hColor=denoise_strength,
            templateWindowSize=7, searchWindowSize=21)
    
    log(f"Saving result to '{output_path}'...")
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        cv2.imwrite(output_path, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(output_path, final_image)


def main():
    """Parses command-line arguments and starts the appropriate stacking process."""
    parser = argparse.ArgumentParser(
        description="Stack video frames to simulate a long exposure. "
                    "By default, an enhancement/denoise filter is applied.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("video_files", nargs='*',
                        help="Path(s) to input video file(s). Ignored if --raw is used.")
    parser.add_argument("-o", "--output", default="max.jpg",
                        help="Path for the final output image. Default: max.jpg")
    parser.add_argument("-s", "--start", type=float, default=0.0,
                        help="Start position in seconds.")
    parser.add_argument("-t", "--duration", type=float, default=None,
                        help="Total duration in seconds to stack from the start time.")
    parser.add_argument("--individual", action="store_true",
                        help="Save a separate stacked image for each input file.")
    parser.add_argument("--quality", type=int, default=95, choices=range(1, 101),
                        metavar="{1-100}", help="Quality for JPEG output (1-100, default: 95).")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                        help="Number of threads for parallel video decoding (default: all cores).")

    raw_group = parser.add_argument_group('Raw Input (from stdin)')
    raw_group.add_argument("--raw", nargs=2, metavar=('W', 'H'), type=int,
                           help="Process raw YUV420 from stdin with specified Width and Height.")
    raw_group.add_argument("--fps", type=float, default=25.0,
                           help="FPS for raw stream timing (default: 25.0).")

    noise_group = parser.add_argument_group('Noise Reduction')
    noise_mutex = noise_group.add_mutually_exclusive_group()
    noise_mutex.add_argument("--denoise", action="store_true",
                             help="Use OpenCV's denoising filter instead of the default enhancement filter.")
    noise_mutex.add_argument("--noenhance", action="store_true",
                             help="Disable the default enhancement filter. No noise reduction will be applied.")
    noise_group.add_argument("--denoise-strength", type=float, default=10.0,
                             help="Strength of the --denoise filter (default: 10.0).")
    
    args = parser.parse_args()

    # By default, enhancement is on. It's disabled if the user explicitly asks
    # for --noenhance or chooses the alternative --denoise filter.
    enhance = not (args.noenhance or args.denoise)

    if args.raw and args.individual:
        parser.error("The --raw and --individual options cannot be used together.")

    if args.raw:
        if args.start > 0 or args.duration is not None:
            print(f"Info: Using FPS of {args.fps} for time calculations in raw mode.", file=sys.stderr)
        width, height = args.raw
        stack_raw_frames(width, height, args.output, args.start, args.duration,
                         args.fps, args.denoise, args.denoise_strength,
                         args.quality, enhance)
    else:
        if not args.video_files:
            parser.error("Missing input. Provide video file(s) or use the --raw option.")
        stack_video_frames(args.video_files, args.output, args.start,
                           args.duration, args.denoise,
                           args.denoise_strength, args.quality, args.threads,
                           args.individual, enhance)

if __name__ == "__main__":
    main()
